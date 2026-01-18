//! Module cache for compiled WASM bytecode
//!
//! Provides two-tier caching (memory + disk) to avoid redundant compilation.

use lru::LruCache;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use tracing::{debug, info, warn};

/// Configuration for the module cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of modules in memory cache
    pub memory_size: usize,
    /// Directory for disk cache (None = memory only)
    pub disk_dir: Option<PathBuf>,
    /// Maximum disk cache size in bytes (0 = unlimited)
    pub max_disk_bytes: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            memory_size: 100,
            disk_dir: None,
            max_disk_bytes: 100 * 1024 * 1024, // 100MB
        }
    }
}

/// Statistics about cache usage
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of entries in memory cache
    pub memory_entries: usize,
    /// Capacity of memory cache
    pub memory_capacity: usize,
    /// Number of entries in disk cache
    pub disk_entries: usize,
    /// Total bytes on disk
    pub disk_bytes: u64,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
}

/// Two-tier cache for compiled WASM modules
pub struct ModuleCache {
    /// In-memory LRU cache: hash -> compiled WASM
    memory: LruCache<String, Vec<u8>>,
    /// Disk cache directory (if enabled)
    disk_dir: Option<PathBuf>,
    /// Whether disk caching is enabled and working
    disk_enabled: bool,
    /// Maximum disk cache size
    max_disk_bytes: u64,
    /// Statistics
    hits: u64,
    misses: u64,
}

impl ModuleCache {
    /// Create a new module cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        let memory = LruCache::new(NonZeroUsize::new(config.memory_size.max(1)).unwrap());

        let (disk_dir, disk_enabled) = if let Some(dir) = config.disk_dir {
            match std::fs::create_dir_all(&dir) {
                Ok(_) => {
                    info!("Disk cache enabled at {:?}", dir);
                    (Some(dir), true)
                }
                Err(e) => {
                    warn!("Failed to create disk cache directory: {}", e);
                    (None, false)
                }
            }
        } else {
            (None, false)
        };

        Self {
            memory,
            disk_dir,
            disk_enabled,
            max_disk_bytes: config.max_disk_bytes,
            hits: 0,
            misses: 0,
        }
    }

    /// Create a memory-only cache with default size
    pub fn memory_only(size: usize) -> Self {
        Self::new(CacheConfig {
            memory_size: size,
            disk_dir: None,
            max_disk_bytes: 0,
        })
    }

    /// Get a cached module by source code
    pub fn get(&mut self, source: &str) -> Option<Vec<u8>> {
        let key = Self::hash_source(source);

        // Check memory cache first
        if let Some(wasm) = self.memory.get(&key) {
            debug!("Cache hit (memory): {}", &key[..8]);
            self.hits += 1;
            return Some(wasm.clone());
        }

        // Check disk cache
        if self.disk_enabled
            && let Some(ref dir) = self.disk_dir
        {
            let disk_path = dir.join(&key);
            if let Ok(wasm) = std::fs::read(&disk_path) {
                debug!("Cache hit (disk): {}", &key[..8]);
                // Promote to memory cache
                self.memory.put(key, wasm.clone());
                self.hits += 1;
                return Some(wasm);
            }
        }

        debug!("Cache miss: {}", &key[..8]);
        self.misses += 1;
        None
    }

    /// Store a compiled module
    pub fn put(&mut self, source: &str, wasm: Vec<u8>) {
        let key = Self::hash_source(source);

        // Write to disk cache first (if enabled)
        if self.disk_enabled
            && let Some(ref dir) = self.disk_dir
        {
            let disk_path = dir.join(&key);
            if let Err(e) = std::fs::write(&disk_path, &wasm) {
                warn!("Failed to write to disk cache: {}", e);
            }
        }

        // Write to memory cache
        debug!("Cached module: {} ({} bytes)", &key[..8], wasm.len());
        self.memory.put(key, wasm);
    }

    /// Check if a module is cached (without loading it)
    pub fn contains(&mut self, source: &str) -> bool {
        let key = Self::hash_source(source);

        if self.memory.contains(&key) {
            return true;
        }

        if self.disk_enabled
            && let Some(ref dir) = self.disk_dir
        {
            return dir.join(&key).exists();
        }

        false
    }

    /// Compute cache key from source code
    fn hash_source(source: &str) -> String {
        let digest = md5::compute(source.as_bytes());
        format!("{:x}.wasm", digest)
    }

    /// Clear all cached modules
    pub fn clear(&mut self) {
        self.memory.clear();
        self.hits = 0;
        self.misses = 0;

        if self.disk_enabled
            && let Some(ref dir) = self.disk_dir
        {
            if let Err(e) = std::fs::remove_dir_all(dir) {
                warn!("Failed to clear disk cache: {}", e);
            } else if let Err(e) = std::fs::create_dir_all(dir) {
                warn!("Failed to recreate disk cache directory: {}", e);
                self.disk_enabled = false;
            }
        }

        info!("Cache cleared");
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let (disk_entries, disk_bytes) = if self.disk_enabled {
            self.disk_stats()
        } else {
            (0, 0)
        };

        CacheStats {
            memory_entries: self.memory.len(),
            memory_capacity: self.memory.cap().get(),
            disk_entries,
            disk_bytes,
            hits: self.hits,
            misses: self.misses,
        }
    }

    /// Get disk cache statistics
    fn disk_stats(&self) -> (usize, u64) {
        if let Some(ref dir) = self.disk_dir {
            match std::fs::read_dir(dir) {
                Ok(entries) => {
                    let mut count = 0;
                    let mut bytes = 0u64;
                    for entry in entries.flatten() {
                        if let Ok(meta) = entry.metadata()
                            && meta.is_file()
                        {
                            count += 1;
                            bytes += meta.len();
                        }
                    }
                    (count, bytes)
                }
                Err(_) => (0, 0),
            }
        } else {
            (0, 0)
        }
    }

    /// Get hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Prune disk cache if it exceeds the size limit
    pub fn prune_disk_cache(&mut self) {
        if !self.disk_enabled || self.max_disk_bytes == 0 {
            return;
        }

        let Some(ref dir) = self.disk_dir else {
            return;
        };

        let (_, current_bytes) = self.disk_stats();
        if current_bytes <= self.max_disk_bytes {
            return;
        }

        info!(
            "Pruning disk cache: {} bytes > {} limit",
            current_bytes, self.max_disk_bytes
        );

        // Get all files with modification times
        let mut files: Vec<(PathBuf, std::time::SystemTime, u64)> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata()
                    && meta.is_file()
                {
                    let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    files.push((entry.path(), mtime, meta.len()));
                }
            }
        }

        // Sort by modification time (oldest first)
        files.sort_by_key(|(_, mtime, _)| *mtime);

        // Delete oldest files until under limit
        let mut bytes_freed = 0u64;
        let target = current_bytes - self.max_disk_bytes;
        for (path, _, size) in files {
            if bytes_freed >= target {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                bytes_freed += size;
                debug!("Pruned: {:?}", path);
            }
        }

        info!("Freed {} bytes from disk cache", bytes_freed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_cache() {
        let mut cache = ModuleCache::memory_only(10);

        let source = "pub fn analyze(input: &str) -> String { input.len().to_string() }";
        let wasm = vec![0x00, 0x61, 0x73, 0x6D]; // WASM magic

        // Should be a miss initially
        assert!(cache.get(source).is_none());
        assert_eq!(cache.stats().misses, 1);

        // Store and retrieve
        cache.put(source, wasm.clone());
        let cached = cache.get(source);
        assert_eq!(cached, Some(wasm));
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_cache_key_consistency() {
        let source1 = "pub fn analyze(input: &str) -> String { \"a\".to_string() }";
        let source2 = "pub fn analyze(input: &str) -> String { \"a\".to_string() }";
        let source3 = "pub fn analyze(input: &str) -> String { \"b\".to_string() }";

        let key1 = ModuleCache::hash_source(source1);
        let key2 = ModuleCache::hash_source(source2);
        let key3 = ModuleCache::hash_source(source3);

        assert_eq!(key1, key2); // Same source = same key
        assert_ne!(key1, key3); // Different source = different key
    }

    #[test]
    fn test_contains() {
        let mut cache = ModuleCache::memory_only(10);
        let source = "test source";
        let wasm = vec![1, 2, 3, 4];

        assert!(!cache.contains(source));
        cache.put(source, wasm);
        assert!(cache.contains(source));
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = ModuleCache::memory_only(2); // Very small cache

        let source1 = "source1";
        let source2 = "source2";
        let source3 = "source3";

        cache.put(source1, vec![1]);
        cache.put(source2, vec![2]);
        cache.put(source3, vec![3]); // Should evict source1

        assert!(cache.get(source1).is_none()); // Evicted
        assert!(cache.get(source2).is_some()); // Still there
        assert!(cache.get(source3).is_some()); // Still there
    }

    #[test]
    fn test_clear() {
        let mut cache = ModuleCache::memory_only(10);

        cache.put("source1", vec![1]);
        cache.put("source2", vec![2]);
        cache.get("source1"); // Create some stats

        cache.clear();

        assert!(cache.get("source1").is_none());
        assert!(cache.get("source2").is_none());
        assert_eq!(cache.stats().memory_entries, 0);
    }

    #[test]
    fn test_stats() {
        let mut cache = ModuleCache::memory_only(10);

        cache.put("source", vec![1, 2, 3, 4]);
        cache.get("source"); // Hit
        cache.get("missing"); // Miss

        let stats = cache.stats();
        assert_eq!(stats.memory_entries, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = ModuleCache::memory_only(10);

        cache.put("source", vec![1]);
        cache.get("source"); // Hit
        cache.get("source"); // Hit
        cache.get("missing"); // Miss

        assert!((cache.hit_rate() - 66.666).abs() < 1.0); // ~66.67%
    }
}
