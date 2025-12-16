#!/usr/bin/env python3
"""
Cache management utility for model predictions.

Allows clearing cache by various properties: backend, model, or clearing everything.
"""

import argparse
import sys
from pathlib import Path

# Add workspace to path if running as script
workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from gpt_oss_utils_dactile.model_predict import (
    cache,
    Model,
    InferenceBackend,
    ModelResponse,
    _get_cache_key,
)


def get_cache_stats():
    """Get statistics about cache contents."""
    stats = {
        'total': 0,
        'by_backend': {},
        'by_model': {},
    }
    
    # Common temperature and max_tokens values to check
    # These are the defaults used in get_model_response
    common_params = [
        (0.0, 8192),  # Default values
        (None, None),  # Legacy without these params
    ]
    
    for key in cache.iterkeys():
        try:
            response = cache.get(key)
            if isinstance(response, ModelResponse):
                stats['total'] += 1
                
                # Count by model
                model_name = str(response.model)
                stats['by_model'][model_name] = stats['by_model'].get(model_name, 0) + 1
                
                # Try to determine backend by checking if key matches
                backend = None
                
                # Try common parameter combinations for both backends
                for temp, max_tok in common_params:
                    # Check LOCAL backend
                    test_key_local = _get_cache_key(
                        model=response.model,
                        prompt=response.prompt,
                        system_prompt=response.system_prompt,
                        temperature=temp,
                        max_tokens=max_tok,
                        backend=InferenceBackend.LOCAL,
                    )
                    if test_key_local == key:
                        backend = 'local'
                        break
                    
                    # Check API backend
                    test_key_api = _get_cache_key(
                        model=response.model,
                        prompt=response.prompt,
                        system_prompt=response.system_prompt,
                        temperature=temp,
                        max_tokens=max_tok,
                        backend=InferenceBackend.API,
                    )
                    if test_key_api == key:
                        backend = 'api'
                        break
                
                # If still not found, it's a legacy entry (no backend in key)
                if backend is None:
                    backend = 'legacy (pre-backend)'
                
                stats['by_backend'][backend] = stats['by_backend'].get(backend, 0) + 1
        except Exception as e:
            # Skip entries that can't be processed
            continue
    
    return stats


def print_cache_stats():
    """Print cache statistics."""
    stats = get_cache_stats()
    
    print(f"\n{'='*60}")
    print(f"Cache Statistics")
    print(f"{'='*60}")
    print(f"Total entries: {stats['total']}")
    print(f"\nBy Backend:")
    for backend, count in sorted(stats['by_backend'].items()):
        print(f"  {backend}: {count}")
    print(f"\nBy Model:")
    for model, count in sorted(stats['by_model'].items()):
        print(f"  {model}: {count}")
    print(f"{'='*60}\n")


def clear_cache_by_backend(backend: InferenceBackend = None):
    """
    Clear cached responses filtered by backend.
    
    Args:
        backend: Backend to clear cache for (LOCAL or API)
    
    Returns:
        int: Number of entries cleared
    """
    keys_to_delete = []
    
    # Common temperature and max_tokens values to check
    common_params = [
        (0.0, 8192),  # Default values
        (None, None),  # Legacy without these params
    ]
    
    for key in cache.iterkeys():
        try:
            response = cache.get(key)
            if isinstance(response, ModelResponse):
                # Try to regenerate the key with the specified backend and common params
                for temp, max_tok in common_params:
                    test_key = _get_cache_key(
                        model=response.model,
                        prompt=response.prompt,
                        system_prompt=response.system_prompt,
                        temperature=temp,
                        max_tokens=max_tok,
                        backend=backend,
                    )
                    if test_key == key:
                        keys_to_delete.append(key)
                        break  # Found a match, no need to check other params
        except Exception:
            continue
    
    # Delete the identified keys
    for key in keys_to_delete:
        del cache[key]
    
    return len(keys_to_delete)


def clear_cache_by_model(model: Model):
    """
    Clear cached responses for a specific model.
    
    Args:
        model: Model to clear cache for
    
    Returns:
        int: Number of entries cleared
    """
    keys_to_delete = []
    
    for key in cache.iterkeys():
        try:
            response = cache.get(key)
            if isinstance(response, ModelResponse) and response.model == model:
                keys_to_delete.append(key)
        except Exception:
            continue
    
    # Delete the identified keys
    for key in keys_to_delete:
        del cache[key]
    
    return len(keys_to_delete)


def clear_all_cache():
    """
    Clear entire cache.
    
    Returns:
        int: Number of entries cleared
    """
    count = len(cache)
    cache.clear()
    return count


def main():
    parser = argparse.ArgumentParser(
        description='Manage model prediction cache',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show cache statistics
  python predict_cache_manage.py --stats
  
  # Clear local backend cache
  python predict_cache_manage.py --clear-backend local
  
  # Clear API backend cache
  python predict_cache_manage.py --clear-backend api
  
  # Clear cache for specific model
  python predict_cache_manage.py --clear-model GPT_OSS_20B
  
  # Clear entire cache
  python predict_cache_manage.py --clear-all
  
  # Clear and show stats
  python predict_cache_manage.py --clear-backend local --stats
        """
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show cache statistics'
    )
    
    parser.add_argument(
        '--clear-backend',
        type=str,
        choices=['local', 'api', 'LOCAL', 'API'],  # Accept both cases for convenience
        help='Clear cache for specific backend (local or api)'
    )
    
    parser.add_argument(
        '--clear-model',
        type=str,
        choices=[m.name for m in Model],
        help='Clear cache for specific model'
    )
    
    parser.add_argument(
        '--clear-all',
        action='store_true',
        help='Clear entire cache (use with caution!)'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show stats by default
    if not any([args.stats, args.clear_backend, args.clear_model, args.clear_all]):
        args.stats = True
    
    # Show stats before clearing (if requested or if clearing)
    if args.stats or args.clear_backend or args.clear_model or args.clear_all:
        print_cache_stats()
    
    # Perform clearing operations
    cleared = 0
    
    if args.clear_all:
        if not args.yes:
            response = input("⚠️  Clear ENTIRE cache? This cannot be undone. [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        
        cleared = clear_all_cache()
        print(f"✓ Cleared entire cache: {cleared} entries")
    
    elif args.clear_backend:
        # Convert to lowercase and then to enum
        backend_str = args.clear_backend.lower()
        backend = InferenceBackend(backend_str)
        
        if not args.yes:
            response = input(f"Clear cache for backend '{backend}'? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        
        cleared = clear_cache_by_backend(backend)
        print(f"✓ Cleared {cleared} entries for backend: {backend}")
    
    elif args.clear_model:
        model = Model[args.clear_model]
        
        if not args.yes:
            response = input(f"Clear cache for model '{model}'? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        
        cleared = clear_cache_by_model(model)
        print(f"✓ Cleared {cleared} entries for model: {model}")
    
    # Show stats after clearing if something was cleared
    if cleared > 0:
        print_cache_stats()


if __name__ == "__main__":
    main()

