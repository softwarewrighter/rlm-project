//! Extract character names and relationship sentences from War and Peace
//!
//! Usage: rustc -O extract-characters.rs -o extract-characters
//!        ./extract-characters < war-and-peace.txt > characters.txt
//!
//! This reduces a 3.3MB novel to ~50KB of relevant character data.

use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    // Relationship words to look for
    let relationship_words: HashSet<&str> = [
        "father", "mother", "son", "daughter", "brother", "sister",
        "wife", "husband", "married", "marry", "wedding", "family",
        "uncle", "aunt", "nephew", "niece", "cousin", "child", "children",
        "parent", "parents", "grandfather", "grandmother", "grandchild",
        "fiancé", "fiancée", "betrothed", "engaged", "heir", "born"
    ].iter().cloned().collect();

    // Noble/important titles
    let titles: HashSet<&str> = [
        "Prince", "Princess", "Count", "Countess", "Duke", "Duchess",
        "Baron", "Baroness", "General", "Colonel", "Captain", "Emperor",
        "Tsar", "Czar", "Marshal", "Monsieur", "Madame", "Mademoiselle"
    ].iter().cloned().collect();

    // Known main character names (to boost recall)
    let main_characters: HashSet<&str> = [
        "Natasha", "Rostov", "Rostova", "Pierre", "Bezukhov", "Bolkonsky",
        "Andrew", "Andrei", "Mary", "Marya", "Nicholas", "Nikolai",
        "Sonya", "Petya", "Boris", "Helene", "Anatole", "Kuragin",
        "Denisov", "Dolokhov", "Kutuzov", "Napoleon", "Bagration"
    ].iter().cloned().collect();

    let mut name_counts: HashMap<String, usize> = HashMap::new();
    let mut relationship_lines: Vec<String> = Vec::new();
    let mut character_lines: Vec<String> = Vec::new();

    // First pass: collect all lines, count names
    let lines: Vec<String> = stdin.lock().lines()
        .filter_map(|l| l.ok())
        .collect();

    for line in &lines {
        let lower = line.to_lowercase();

        // Check for relationship words
        let has_relationship = relationship_words.iter().any(|w| lower.contains(w));

        // Check for titles
        let has_title = titles.iter().any(|t| line.contains(t));

        // Check for main character names
        let has_main_char = main_characters.iter().any(|c| line.contains(c));

        // Extract capitalized words (potential names)
        for word in line.split_whitespace() {
            let clean: String = word.chars()
                .filter(|c| c.is_alphabetic())
                .collect();
            if clean.len() >= 3 && clean.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                // Skip common words
                if !["The", "And", "But", "For", "That", "This", "With", "From", "Have", "Was", "Were", "His", "Her", "She", "They", "What", "When", "Where", "Which", "Would", "Could", "Should", "There", "Their", "About", "After", "Before", "Between", "Through", "During", "Without", "Against", "Among", "Within", "Along", "Around", "Behind", "Below", "Beneath", "Beside", "Beyond", "Down", "Into", "Near", "Off", "Onto", "Out", "Over", "Past", "Since", "Than", "Toward", "Under", "Until", "Upon", "While", "Chapter", "Part", "Book", "BOOK", "CHAPTER", "PART", "EPILOGUE", "First", "Second"].contains(&clean.as_str()) {
                    *name_counts.entry(clean).or_insert(0) += 1;
                }
            }
        }

        // Collect relevant lines
        if has_relationship && (has_title || has_main_char) {
            relationship_lines.push(line.clone());
        } else if has_title && has_main_char {
            character_lines.push(line.clone());
        }
    }

    // Output section 1: Main characters by frequency
    writeln!(stdout, "=== MAIN CHARACTERS (by frequency) ===\n").unwrap();
    let mut sorted_names: Vec<_> = name_counts.iter()
        .filter(|(_, &count)| count >= 20)  // Appear at least 20 times
        .collect();
    sorted_names.sort_by(|a, b| b.1.cmp(a.1));

    for (name, count) in sorted_names.iter().take(100) {
        writeln!(stdout, "{}: {} mentions", name, count).unwrap();
    }

    // Output section 2: Relationship sentences
    writeln!(stdout, "\n=== RELATIONSHIP SENTENCES ===\n").unwrap();

    // Deduplicate and limit
    let mut seen: HashSet<String> = HashSet::new();
    let mut count = 0;
    for line in &relationship_lines {
        let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if !seen.contains(&normalized) && count < 500 {
            writeln!(stdout, "{}", line.trim()).unwrap();
            seen.insert(normalized);
            count += 1;
        }
    }

    // Output section 3: Character context sentences
    writeln!(stdout, "\n=== CHARACTER CONTEXT ===\n").unwrap();
    seen.clear();
    count = 0;
    for line in &character_lines {
        let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if !seen.contains(&normalized) && count < 300 {
            writeln!(stdout, "{}", line.trim()).unwrap();
            seen.insert(normalized);
            count += 1;
        }
    }

    eprintln!("Extracted {} relationship sentences, {} character sentences",
              relationship_lines.len(), character_lines.len());
}
