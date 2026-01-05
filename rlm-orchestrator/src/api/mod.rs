//! REST API for RLM orchestrator

use crate::orchestrator::{RlmOrchestrator, RlmResult};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// API state
pub struct ApiState {
    pub orchestrator: Arc<RlmOrchestrator>,
}

/// Request to process a query
#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    /// The query to process
    pub query: String,
    /// The context to analyze
    pub context: String,
}

/// Response from a query
#[derive(Debug, Serialize)]
pub struct QueryResponse {
    /// Whether the query succeeded
    pub success: bool,
    /// The answer (if successful)
    pub answer: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Number of iterations taken
    pub iterations: usize,
    /// Total sub-LM calls made
    pub sub_calls: usize,
}

impl From<RlmResult> for QueryResponse {
    fn from(result: RlmResult) -> Self {
        Self {
            success: result.success,
            answer: if result.success {
                Some(result.answer)
            } else {
                None
            },
            error: result.error,
            iterations: result.iterations,
            sub_calls: result.total_sub_calls,
        }
    }
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// Create the API router
pub fn create_router(state: Arc<ApiState>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/query", post(process_query))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Health check endpoint
async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Process a query
async fn process_query(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    match state.orchestrator.process(&request.query, &request.context).await {
        Ok(result) => Ok(Json(result.into())),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}
