//! Python REPL integration via PyO3

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use thiserror::Error;

/// Errors from REPL execution
#[derive(Error, Debug)]
pub enum ReplError {
    #[error("Python error: {0}")]
    PythonError(String),

    #[error("Execution timeout")]
    Timeout,

    #[error("Code extraction failed: {0}")]
    ExtractionError(String),
}

/// Python REPL for code execution
pub struct PythonRepl {
    /// Persistent context store (variables that persist across executions)
    context_store: HashMap<String, PyObject>,
}

impl PythonRepl {
    /// Create a new Python REPL
    pub fn new() -> Self {
        Self {
            context_store: HashMap::new(),
        }
    }

    /// Execute Python code and return the output
    pub fn execute(&mut self, code: &str, context: &str) -> Result<String, ReplError> {
        Python::with_gil(|py| {
            // Create the execution namespace
            let globals = PyDict::new(py);
            let locals = PyDict::new(py);

            // Inject the context
            locals
                .set_item("context", context)
                .map_err(|e| ReplError::PythonError(e.to_string()))?;

            // Inject persistent context store
            for (key, value) in &self.context_store {
                locals
                    .set_item(key.as_str(), value)
                    .map_err(|e| ReplError::PythonError(e.to_string()))?;
            }

            // Import common modules
            py.run(
                c"import re, json, collections, itertools, functools",
                Some(&globals),
                Some(&locals),
            )
            .map_err(|e| ReplError::PythonError(e.to_string()))?;

            // Capture stdout
            py.run(
                c"import io, sys; _stdout = io.StringIO(); sys.stdout = _stdout",
                Some(&globals),
                Some(&locals),
            )
            .map_err(|e| ReplError::PythonError(e.to_string()))?;

            // Execute the code
            let exec_result = py.run(
                &std::ffi::CString::new(code).unwrap(),
                Some(&globals),
                Some(&locals),
            );

            // Capture the output
            let output = py
                .eval(c"_stdout.getvalue()", Some(&globals), Some(&locals))
                .map_err(|e| ReplError::PythonError(e.to_string()))?
                .extract::<String>()
                .map_err(|e| ReplError::PythonError(e.to_string()))?;

            // Reset stdout
            py.run(
                c"sys.stdout = sys.__stdout__",
                Some(&globals),
                Some(&locals),
            )
            .ok();

            // Handle execution errors
            if let Err(e) = exec_result {
                return Err(ReplError::PythonError(format!(
                    "{}\nOutput before error: {}",
                    e, output
                )));
            }

            // Update context store with new variables
            for item in locals.items().iter() {
                if let Ok((key, value)) = item.extract::<(String, PyObject)>() {
                    // Skip internal variables
                    if !key.starts_with('_') && key != "context" {
                        self.context_store.insert(key, value);
                    }
                }
            }

            Ok(output)
        })
    }

    /// Get a variable from the context store
    pub fn get_variable(&self, name: &str) -> Option<String> {
        Python::with_gil(|py| {
            self.context_store
                .get(name)
                .and_then(|obj| obj.extract::<String>(py).ok())
        })
    }

    /// Clear the context store
    pub fn clear(&mut self) {
        self.context_store.clear();
    }
}

impl Default for PythonRepl {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract Python code blocks from LLM response
pub fn extract_code(response: &str) -> Option<String> {
    // Look for ```repl or ```python blocks
    let patterns = [
        (r"```repl\n", r"```"),
        (r"```python\n", r"```"),
    ];

    for (start_pat, end_pat) in &patterns {
        if let Some(start_idx) = response.find(start_pat) {
            let code_start = start_idx + start_pat.len();
            if let Some(end_idx) = response[code_start..].find(end_pat) {
                return Some(response[code_start..code_start + end_idx].to_string());
            }
        }
    }

    None
}

/// Extract FINAL answer from LLM response
pub fn extract_final(response: &str) -> Option<String> {
    // FINAL(answer) pattern
    if let Some(start) = response.find("FINAL(") {
        let content_start = start + 6;
        if let Some(end) = response[content_start..].find(')') {
            return Some(response[content_start..content_start + end].to_string());
        }
    }

    // FINAL_VAR(variable_name) pattern - need to look up variable
    if let Some(start) = response.find("FINAL_VAR(") {
        let content_start = start + 10;
        if let Some(end) = response[content_start..].find(')') {
            let var_name = &response[content_start..content_start + end];
            return Some(format!("__FINAL_VAR__{}", var_name));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code() {
        let response = "Let me analyze this:\n```python\nprint('hello')\n```\nDone.";
        assert_eq!(extract_code(response), Some("print('hello')\n".to_string()));
    }

    #[test]
    fn test_extract_final() {
        assert_eq!(
            extract_final("FINAL(The answer is 42)"),
            Some("The answer is 42".to_string())
        );
        assert_eq!(
            extract_final("FINAL_VAR(result)"),
            Some("__FINAL_VAR__result".to_string())
        );
    }
}
