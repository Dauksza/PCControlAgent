"""
Cache manager for storing temporary data
"""
import json
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from pathlib import Path

class CacheManager:
    """
    Simple file-based cache manager with TTL support
    Can be extended to use Redis in production
    """
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if self._is_valid(entry):
                return entry["data"]
            else:
                del self.memory_cache[key]
        
        # Check file cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)
                
                if self._is_valid(entry):
                    # Load into memory cache
                    self.memory_cache[key] = entry
                    return entry["data"]
                else:
                    cache_file.unlink()
            except Exception:
                pass
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        entry = {
            "data": value,
            "timestamp": datetime.now().isoformat(),
            "ttl": ttl
        }
        
        # Store in memory
        self.memory_cache[key] = entry
        
        # Store in file
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(entry, f)
        except Exception:
            pass
    
    def delete(self, key: str):
        """Delete value from cache"""
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()
    
    def clear(self):
        """Clear all cache"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
    
    def _is_valid(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        if "ttl" not in entry or entry["ttl"] is None:
            return True
        
        timestamp = datetime.fromisoformat(entry["timestamp"])
        elapsed = (datetime.now() - timestamp).total_seconds()
        
        return elapsed < entry["ttl"]
