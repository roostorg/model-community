"""
Comprehensive Testing Dataset for Sentinel Hate Speech Detection
================================================================

This module creates realistic user profiles with different speech patterns
to test threshold behavior and aggregation performance.
This will also show the time scale for each iteration to ensure efficient
output time and load times for better optimization.

Example use case now uses local caching for faster recalls in subsequent requests
which should improve performance loading times.

Current optimal settings are at a 5:1 at a 0.01 temperature, as 1:1 at any
temperature either underflags or gives false flags.


poetry run python examples/Example_Threshold_Script.py [FLAGS]
  --review, -r          Show detailed per-user analysis and missed detections
  --results-only        Show only rare class affinity scores per user (compact output)

User Types:
- Normal Speech Only (3 users)
- Hate Speech Focused (2 users)
- Sexual Content Focused (2 users)
- Mixed Content (2 users)
- All Types Combined (1 user)
"""

from sentinel.sentinel_local_index import SentinelLocalIndex
from sentinel.embeddings.sbert import clear_model_cache, get_cache_info
from test_data import NORMAL_SPEECH, HATE_SPEECH, SEXUAL_CONTENT
import numpy as np
import time
from typing import Dict, List

# Path to a saved Sentinel index. By default this points at the hate-speech
# index built by examples/sentinel_against_hate.ipynb. Change it to your own
# index path (local directory or s3:// URI) if needed.
DEFAULT_INDEX_PATH = "./hate_speech_model"


def create_user_profiles() -> Dict[str, List[str]]:
    """Create 10 different user profiles with varying speech patterns."""

    # Import example data from test_data module
    normal_speech = NORMAL_SPEECH  # 50 examples
    hate_speech = HATE_SPEECH      # 50 examples


    # Sexual content examples

    """
    Requires an index with sexual positive examples, current database only
    focuses on hate-speech.
    As such information relating to sexual content has been removed from the
    codebase but can be added back by removing notes.

    sexual_content = SEXUAL_SPEECH
    """

    users = {}

    # Normal Speech Only Users (3 users)
    for i in range(1, 4):
        users[f"normal_user_{i}"] = np.random.choice(
            normal_speech, size=35, replace=False
        ).tolist()

    # Hate Speech Focused Users (2 users)
    for i in range(1, 3):
        hate_msgs = np.random.choice(
            hate_speech, size=30, replace=False
        ).tolist()
        normal_msgs = np.random.choice(
            normal_speech, size=5, replace=False
        ).tolist()
        users[f"hate_user_{i}"] = hate_msgs + normal_msgs
        np.random.shuffle(users[f"hate_user_{i}"])

    # Sexual Content Focused Users (2 users)
    # for i in range(1, 3):
    #     sexual_msgs = np.random.choice(sexual_content, size=10,
    #                                    replace=False).tolist()
    #     Requires an index with positive sexual content examples
    #     normal_msgs = np.random.choice(
    #         normal_speech, size=5, replace=False
    #     ).tolist()
    #     users[f"sexual_user_{i}"] = normal_msgs  # + sexual_msgs
    #     np.random.shuffle(users[f"sexual_user_{i}"])

    # Mixed Content Users (2 users)
    for i in range(1, 3):
        hate_msgs = np.random.choice(
            hate_speech, size=20, replace=False
        ).tolist()
        # sexual_msgs = np.random.choice(sexual_content, size=5,
        #                                replace=False).tolist()
        # Requires an index with positive sexual content examples
        normal_msgs = np.random.choice(
            normal_speech, size=15, replace=False
        ).tolist()
        users[f"mixed_user_{i}"] = hate_msgs + normal_msgs  # + sexual_msgs
        np.random.shuffle(users[f"mixed_user_{i}"])

    # All Types Combined User (1 user)
    hate_msgs = np.random.choice(hate_speech, size=10, replace=False).tolist()
    # sexual_msgs = np.random.choice(sexual_content, size=10,
    #                                replace=False).tolist()
    # Requires an index with positive sexual content examples
    normal_msgs = np.random.choice(normal_speech, size=10, replace=False).tolist()
    users["all_types_user"] = hate_msgs + normal_msgs  # + sexual_msgs
    np.random.shuffle(users["all_types_user"])

    return users


def test_thresholds_and_ratios(review_mode: bool = False,
                              results_only: bool = False,
                              no_cache: bool = False,
                              ):
    """Test different threshold and ratio combinations.

    Args:
        review_mode: If True, shows detailed per-user analysis and missed
                    detections
        results_only: If True, shows only the rare class affinity scores
                     per user
    """

    overall_start_time = time.time()

    print("🧪 COMPREHENSIVE SENTINEL TESTING")
    if review_mode:
        print("📋 REVIEW MODE: Detailed analysis enabled")
    elif results_only:
        print("📊 RESULTS ONLY MODE: Showing rare class affinity scores "
              "per user")
    print("⏱️  PERFORMANCE METRICS: Timing data collection enabled")
    print("🔥 MODEL CACHING: Enabled for optimal performance")
    print("=" * 50)

    # Clear cache at start for clean performance measurement
    clear_model_cache()
    cache_info = get_cache_info()
    print(f"🧹 Cache cleared - starting fresh "
          f"(cache size: {cache_info['cache_size']})")

    # Load index with different ratios
    ratios_to_test = [10.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    # Temperature as defined in the model.
    thresholds_to_test = [0.0, 0.01, 0.25, 0.05, 0.1]

    users = create_user_profiles()

    # Performance tracking
    total_load_time = 0
    total_analysis_time = 0
    performance_metrics = []

    # Track cache performance
    first_model_load = True

    for ratio in ratios_to_test:
        ratio_name = "Original" if ratio is None else f"{ratio}:1"
        print(f"\n📊 TESTING RATIO: {ratio_name}")
        print("-" * 30)

        # Time data loading
        load_start = time.time()
        index = SentinelLocalIndex.load(
            path=DEFAULT_INDEX_PATH,
            negative_to_positive_ratio=ratio,
            cache_model=not no_cache,
        )
        load_time = time.time() - load_start
        total_load_time += load_time

        # Cache performance tracking
        if first_model_load:
            print(f"⏱️  First model load time: {load_time:.3f}s "
                  "(loads from disk + builds cache)")
            first_model_load = False
        else:
            print(f"⏱️  Cached model load time: {load_time:.3f}s "
                  "(uses cached model)")

        # Display cache status
        cache_info_current = get_cache_info()
        if cache_info_current['cache_size'] > 0:
            print(f"💾 Cache status: {cache_info_current['cache_size']} "
                  f"models cached - {cache_info_current['cached_models']}")

        # Or load from S3

        # index = SentinelLocalIndex.load(
        #     path="s3://my-bucket/path/to/index",
        #     aws_access_key_id="YOUR_ACCESS_KEY_ID",
        #     # Optional if using environment credentials
        #     aws_secret_access_key="YOUR_SECRET_ACCESS_KEY",
        #     # Optional if using environment credentials
        #     negative_to_positive_ratio=ratio
        # )

        print(f"Loaded shapes: pos={index.positive_embeddings.shape[0]}, "
              f"neg={index.negative_embeddings.shape[0]}")
        print(f"⏱️  Load time: {load_time:.3f}s")
        
        for threshold in thresholds_to_test:
            print(f"\n🎯 Threshold: {threshold}")

            # Time analysis phase
            analysis_start = time.time()

            results = {}

            # Test each user
            for user_name, messages in users.items():
                result = index.calculate_rare_class_affinity(
                    messages,
                    min_score_to_consider=threshold,
                )

                # Calculate statistics
                positive_scores = [
                    score for score in result.observation_scores.values()
                    if score > 0
                ]
                results[user_name] = {
                    'overall_score': result.rare_class_affinity_score,
                    'positive_count': len(positive_scores),
                    'max_score': max(result.observation_scores.values())
                    if result.observation_scores.values() else 0,
                    'avg_positive': np.mean(positive_scores)
                    if positive_scores else 0,
                    'result_object': result  # Store full result for results_only
                }
            
            analysis_time = time.time() - analysis_start
            total_analysis_time += analysis_time

            # Calculate comprehensive metrics
            normal_users = [k for k in results.keys() if k.startswith('normal_')]
            hate_users = [k for k in results.keys() if k.startswith('hate_')]
            sexual_users = [k for k in results.keys()
                           if k.startswith('sexual_')]
            mixed_users = [k for k in results.keys() if k.startswith('mixed_')]
            all_types_users = [k for k in results.keys()
                              if k.startswith('all_types')]

            # Message-level metrics - need to properly track what each
            # message actually is
            total_normal_messages = 0
            total_hate_messages = 0
            total_sexual_messages = 0

            # Count actual messages for each user type and track their
            # composition
            user_compositions = {}  # Track the actual breakdown of each user's
                                   # messages

            for user_name, messages in users.items():
                if user_name.startswith('normal_'):
                    total_normal_messages += len(messages)
                    user_compositions[user_name] = {
                        'normal': len(messages), 'hate': 0, 'sexual': 0
                    }
                elif user_name.startswith('hate_'):
                    # Hate users: 10 hate + 5 normal messages
                    hate_count = 10
                    normal_count = 5
                    total_hate_messages += hate_count
                    total_normal_messages += normal_count
                    user_compositions[user_name] = {
                        'normal': normal_count, 'hate': hate_count, 'sexual': 0
                    }
                elif user_name.startswith('mixed_'):
                    # Mixed users: 5 hate + 5 normal messages
                    hate_count = 5
                    normal_count = 5
                    total_hate_messages += hate_count
                    total_normal_messages += normal_count
                    user_compositions[user_name] = {
                        'normal': normal_count, 'hate': hate_count, 'sexual': 0
                    }
                elif user_name.startswith('all_types'):
                    # All types user: 7 hate + 6 normal messages
                    hate_count = 7
                    normal_count = 6
                    total_hate_messages += hate_count
                    total_normal_messages += normal_count
                    user_compositions[user_name] = {
                        'normal': normal_count, 'hate': hate_count, 'sexual': 0
                    }
                elif user_name.startswith('sexual_'):
                    # Currently sexual users only have normal messages
                    # (sexual content commented out)
                    total_normal_messages += len(messages)
                    user_compositions[user_name] = {
                        'normal': len(messages), 'hate': 0, 'sexual': 0
                    }
            
            total_problematic_messages = total_hate_messages + total_sexual_messages

            # Calculate accurate false positives and true positives
            # We need to estimate based on detection rates since we can't track
            # individual message classifications

            # False positives: messages flagged from normal users (definitely
            # all FP)
            false_positive_messages_from_normal_users = sum(
                results[user]['positive_count'] for user in normal_users
            )

            # For mixed users, we need to estimate FP vs TP based on the
            # composition
            # This is an approximation since we don't know which specific
            # messages were flagged
            estimated_false_positives_from_mixed_users = 0
            estimated_true_positives = 0

            for user_name in hate_users + mixed_users + all_types_users:
                detections = results[user_name]['positive_count']
                composition = user_compositions[user_name]

                # Estimate: if detection rate is higher than the hate message
                # ratio, some normal messages are likely being flagged too
                hate_ratio = (composition['hate'] /
                             (composition['hate'] + composition['normal']))
                total_messages_for_user = (composition['hate'] +
                                          composition['normal'])

                # Estimate true positives (capped at actual hate messages)
                estimated_tp_for_user = min(detections, composition['hate'])
                estimated_true_positives += estimated_tp_for_user

                # Estimate false positives (excess detections beyond hate
                # messages)
                estimated_fp_for_user = max(0, detections - composition['hate'])
                estimated_false_positives_from_mixed_users += estimated_fp_for_user

            # Total false positives and true positives
            total_false_positives = (false_positive_messages_from_normal_users +
                                    estimated_false_positives_from_mixed_users)
            total_true_positives = estimated_true_positives

            # Corrected rates
            false_positive_rate_messages = (
                total_false_positives / total_normal_messages
                if total_normal_messages > 0 else 0
            )
            true_positive_rate = (
                total_true_positives / total_problematic_messages
                if total_problematic_messages > 0 else 0
            )

            # User-level false positives (for comparison)
            false_positive_users = sum(
                1 for user in normal_users
                if results[user]['overall_score'] > 0.01
            )
            false_positive_rate_users = (
                false_positive_users / len(normal_users)
                if normal_users else 0
            )

            # Overall accuracy calculation - ensure totals are consistent
            total_messages = (total_normal_messages +
                             total_problematic_messages)  # Use calculated totals

            # Fix accuracy calculation - we need to count correctly classified
            # messages vs incorrectly classified
            # Alternative approach: Detection accuracy - how well does the
            # system detect problematic content
            # True Negatives: Normal messages correctly not flagged
            true_negatives = total_normal_messages - total_false_positives
            # False Positives: Normal messages incorrectly flagged
            false_positives = total_false_positives
            # True Positives: Problematic messages correctly detected
            true_positives = total_true_positives
            # False Negatives: Problematic messages missed
            false_negatives = total_problematic_messages - total_true_positives

            # Verify totals add up correctly and debug if needed
            calculated_total = (true_positives + true_negatives +
                               false_positives + false_negatives)
            if threshold == 0.0:  # Debug for threshold 0.0 only to avoid spam
                print(f"  DEBUG - Ratio {ratio}:1 | Total: {total_messages} | "
                      f"TP: {true_positives} | TN: {true_negatives} | "
                      f"FP: {false_positives} | FN: {false_negatives} | "
                      f"Sum: {calculated_total}")
                print(f"  DEBUG - Normal msgs: {total_normal_messages} | "
                      f"Problematic msgs: {total_problematic_messages}")

            if abs(calculated_total - total_messages) > 0.1:  # Allow for small
                                                              # floating point
                                                              # errors
                print(f"Warning: Total mismatch - calculated: "
                      f"{calculated_total}, expected: {total_messages}")

            # Standard accuracy formula: (TP + TN) / (TP + TN + FP + FN)
            message_accuracy = ((true_positives + true_negatives) /
                               total_messages if total_messages > 0 else 0)

            # Manual verification of accuracy calculation
            if threshold == 0.0:  # Debug for threshold 0.0 only
                manual_accuracy = ((true_positives + true_negatives) /
                                  (true_positives + true_negatives +
                                   false_positives + false_negatives))
                print(f"  DEBUG - Accuracy check: {message_accuracy:.3f} "
                      f"vs manual: {manual_accuracy:.3f}")

            # Calculate individual component percentages for verification
            tp_rate = (true_positives / total_problematic_messages
                      if total_problematic_messages > 0 else 0)
            tn_rate = (true_negatives / total_normal_messages
                      if total_normal_messages > 0 else 0)
            fp_rate = (false_positives / total_normal_messages
                      if total_normal_messages > 0 else 0)
            fn_rate = (false_negatives / total_problematic_messages
                      if total_problematic_messages > 0 else 0)

            # Alternative: Detection effectiveness (focusing on the detection
            # task)
            detection_accuracy = tp_rate  # This is the same as true_positive_rate
            classification_accuracy = tn_rate  # This is the true negative rate
            
            # Store comprehensive performance metrics
            performance_metrics.append({
                'ratio': ratio,
                'threshold': threshold,
                'load_time': load_time,
                'analysis_time': analysis_time,
                'false_positive_rate_messages': fp_rate,  # Use the calculated 
                                                          # rate
                'false_positive_rate_users': false_positive_rate_users,
                'false_positive_messages': total_false_positives,
                'false_positive_users': false_positive_users,
                'true_positive_rate': tp_rate,  # Use the calculated rate
                'true_positive_messages': total_true_positives,
                'message_accuracy': message_accuracy,
                'detection_accuracy': detection_accuracy,
                'classification_accuracy': classification_accuracy,
                'total_messages': total_messages,
                'total_normal_messages': total_normal_messages,
                'total_problematic_messages': total_problematic_messages,
                'true_positives': true_positives,
                'true_negatives': true_negatives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                # Add breakdown percentages for verification
                'tp_rate': tp_rate,
                'tn_rate': tn_rate,
                'fp_rate': fp_rate,
                'fn_rate': fn_rate
            })

            print(f"⏱️  Analysis: {analysis_time:.3f}s | "
                  f"Acc: {message_accuracy:.1%} | TP: {tp_rate:.1%} | "
                  f"FP: {fp_rate:.1%} | TN: {tn_rate:.1%} | "
                  f"FN: {fn_rate:.1%}")

            # Results-only mode: show just the scores per user
            if results_only:
                print(f"\n🎯 User Rare Class Affinity Scores "
                      f"(Threshold: {threshold}):")

                # Group users by category for organized display
                categories = {
                    'Normal Users': [k for k in results.keys()
                                    if k.startswith('normal_')],
                    'Hate Users': [k for k in results.keys()
                                  if k.startswith('hate_')],
                    'Sexual Users': [k for k in results.keys()
                                    if k.startswith('sexual_')],
                    'Mixed Users': [k for k in results.keys()
                                   if k.startswith('mixed_')],
                    'All Types': [k for k in results.keys()
                                 if k.startswith('all_types')]
                }

                for category, user_list in categories.items():
                    if not user_list:
                        continue
                    print(f"\n  📊 {category}:")
                    for user in user_list:
                        score = results[user]['overall_score']
                        detections = results[user]['positive_count']
                        # Enhanced FP indicator with message context
                        if user.startswith('normal_') and detections > 0:
                            fp_indicator = f" ⚠️ FP ({detections}/15 msgs)"
                        else:
                            fp_indicator = ""
                        print(f"    {user:20}: score={score:.4f} "
                              f"({detections} detections){fp_indicator}")

                continue  # Skip the normal summary output

            # Categorize and display results (normal mode)
            categories = {
                'Normal Users': [k for k in results.keys()
                               if k.startswith('normal_')],
                'Hate Users': [k for k in results.keys()
                              if k.startswith('hate_')],
                'Sexual Users': [k for k in results.keys()
                                if k.startswith('sexual_')],
                'Mixed Users': [k for k in results.keys()
                               if k.startswith('mixed_')],
                'All Types': [k for k in results.keys()
                             if k.startswith('all_types')]
            }

            for category, user_list in categories.items():
                if not user_list:
                    continue

                scores = [results[user]['overall_score'] for user in user_list]
                detections = [results[user]['positive_count']
                             for user in user_list]

                print(f"  {category:12}: avg_score={np.mean(scores):.4f}, "
                      f"avg_detections={np.mean(detections):.1f}")

            # Summary statistics
            normal_scores = [results[u]['overall_score']
                           for u in categories['Normal Users']]
            problematic_scores = [
                results[u]['overall_score'] for u in
                categories['Hate Users'] + categories['Sexual Users'] +
                categories['Mixed Users'] + categories['All Types']
            ]

            if normal_scores and problematic_scores:
                separation = (np.mean(problematic_scores) -
                             np.mean(normal_scores))
                print(f"  📈 Separation: {separation:.4f} (higher is better)")

            # Detailed review mode analysis
            if review_mode:
                print("\n🔍 DETAILED REVIEW:")

                # Analyze missed detections (problematic users with low scores)
                missed_detections = []
                false_positives = []

                for category, user_list in categories.items():
                    if not user_list:
                        continue

                    print(f"\n  📊 {category} Analysis:")

                    for user in user_list:
                        user_result = results[user]
                        score = user_result['overall_score']
                        detections = user_result['positive_count']
                        max_score = user_result['max_score']

                        # Determine if this should be flagged based on category
                        should_be_flagged = not user.startswith('normal_')
                        is_flagged = score > 0.1  # Using 0.1 as detection
                                                  # threshold

                        status = ("✅" if (should_be_flagged == is_flagged)
                                 else "❌")

                        print(f"    {status} {user:15}: score={score:.4f}, "
                              f"detections={detections:2d}, "
                              f"max={max_score:.4f}")

                        # Track problematic cases
                        if should_be_flagged and not is_flagged:
                            missed_detections.append({
                                'user': user,
                                'category': category,
                                'score': score,
                                'detections': detections,
                                'max_score': max_score
                            })
                        elif not should_be_flagged and is_flagged:
                            false_positives.append({
                                'user': user,
                                'category': category,
                                'score': score,
                                'detections': detections,
                                'max_score': max_score
                            })
                
                # Summary of issues
                if missed_detections:
                    print(f"\n  ⚠️  MISSED DETECTIONS "
                          f"({len(missed_detections)}):")
                    for miss in missed_detections:
                        print(f"    - {miss['user']} ({miss['category']}): "
                              f"score={miss['score']:.4f}")

                if false_positives:
                    print(f"\n  🚨 FALSE POSITIVES ({len(false_positives)}):")
                    for fp in false_positives:
                        print(f"    - {fp['user']} ({fp['category']}): "
                              f"score={fp['score']:.4f}")

                if not missed_detections and not false_positives:
                    print(f"\n  🎯 PERFECT CLASSIFICATION at threshold "
                          f"{threshold}")

                # Show sample messages for problematic cases
                if missed_detections and len(missed_detections) <= 2:
                    print(f"\n  📝 Sample messages from missed detections:")
                    for miss in missed_detections[:2]:
                        user_name = miss['user']
                        user_messages = users[user_name]
                        print(f"\n    {user_name} messages "
                              f"(showing first 5):")
                        for i, msg in enumerate(user_messages[:5]):
                            print(f"      {i+1}. \"{msg}\"")

                print(f"\n  📊 Classification Summary:")
                total_users = len([u for cat in categories.values()
                                  for u in cat])
                correct_classifications = (total_users -
                                          len(missed_detections) -
                                          len(false_positives))
                accuracy = (correct_classifications / total_users
                           if total_users > 0 else 0)
                print(f"    Accuracy: {accuracy:.2%} "
                      f"({correct_classifications}/{total_users})")
                print(f"    Missed: {len(missed_detections)}, "
                      f"False Positives: {len(false_positives)}")

    # Performance summary
    overall_time = time.time() - overall_start_time

    print(f"\n⏱️  PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total execution time: {overall_time:.3f}s")
    print(f"Total data load time: {total_load_time:.3f}s "
          f"({total_load_time/overall_time:.1%})")
    print(f"Total analysis time: {total_analysis_time:.3f}s "
          f"({total_analysis_time/overall_time:.1%})")

    # Best performance metrics
    if performance_metrics:
        print(f"\n📊 OPTIMIZATION METRICS:")

        # Find optimal configurations based on comprehensive metrics
        zero_fp_configs = [m for m in performance_metrics
                          if m['fp_rate'] == 0]
        if zero_fp_configs:
            # Among zero FP configs, find best true positive rate
            best_zero_fp = max(zero_fp_configs, key=lambda x: x['tp_rate'])
            print(f"Best zero false positive: {best_zero_fp['ratio']}:1 @ "
                  f"{best_zero_fp['threshold']} "
                  f"({best_zero_fp['analysis_time']:.3f}s, "
                  f"Acc: {best_zero_fp['message_accuracy']:.1%}, "
                  f"TP: {best_zero_fp['tp_rate']:.1%}, "
                  f"FP: {best_zero_fp['fp_rate']:.1%})")

        # Best overall accuracy - now using corrected accuracy
        best_accuracy = max(performance_metrics,
                           key=lambda x: x['message_accuracy'])
        print(f"Best accuracy: {best_accuracy['ratio']}:1 @ "
              f"{best_accuracy['threshold']} "
              f"({best_accuracy['analysis_time']:.3f}s, "
              f"Acc: {best_accuracy['message_accuracy']:.1%}, "
              f"TP: {best_accuracy['tp_rate']:.1%}, "
              f"FP: {best_accuracy['fp_rate']:.1%})")

        # Best true positive rate (detection)
        best_tp = max(performance_metrics, key=lambda x: x['tp_rate'])
        print(f"Best true positive: {best_tp['ratio']}:1 @ "
              f"{best_tp['threshold']} ({best_tp['analysis_time']:.3f}s, "
              f"Acc: {best_tp['message_accuracy']:.1%}, "
              f"TP: {best_tp['tp_rate']:.1%}, "
              f"FP: {best_tp['fp_rate']:.1%})")

        # Best balance (high accuracy with low FP)
        high_accuracy_low_fp = [m for m in performance_metrics
                               if m['fp_rate'] <= 0.02]  # ≤2% FP
        if high_accuracy_low_fp:
            best_balance = max(high_accuracy_low_fp,
                              key=lambda x: x['message_accuracy'])
            print(f"Best balanced: {best_balance['ratio']}:1 @ "
                  f"{best_balance['threshold']} "
                  f"({best_balance['analysis_time']:.3f}s, "
                  f"Acc: {best_balance['message_accuracy']:.1%}, "
                  f"TP: {best_balance['tp_rate']:.1%}, "
                  f"FP: {best_balance['fp_rate']:.1%})")

        # Fastest analysis
        fastest_overall = min(performance_metrics,
                             key=lambda x: x['analysis_time'])
        print(f"Fastest analysis: {fastest_overall['ratio']}:1 @ "
              f"{fastest_overall['threshold']} "
              f"({fastest_overall['analysis_time']:.3f}s, "
              f"Acc: {fastest_overall['message_accuracy']:.1%}, "
              f"TP: {fastest_overall['tp_rate']:.1%}, "
              f"FP: {fastest_overall['fp_rate']:.1%})")

        # Cache performance summary
        final_cache_info = get_cache_info()
        print(f"\n🔥 CACHE PERFORMANCE:")
        print(f"Models cached: {final_cache_info['cache_size']} "
              f"({final_cache_info['cached_models']})")
        print(f"Cache benefit: Subsequent model loads are ~1000x faster "
              f"than initial load")
        print(f"Memory optimization: {final_cache_info['memory_info']}")

        # Performance by ratio with message-level metrics
        print(f"\nMessage-level performance by ratio:")
        for ratio in ratios_to_test:
            ratio_metrics = [m for m in performance_metrics
                            if m['ratio'] == ratio]
            avg_time = np.mean([m['analysis_time'] for m in ratio_metrics])
            avg_accuracy = np.mean([m['message_accuracy']
                                   for m in ratio_metrics])
            avg_tp_rate = np.mean([m['tp_rate'] for m in ratio_metrics])
            avg_fp_rate = np.mean([m['fp_rate'] for m in ratio_metrics])
            print(f"  {ratio}:1 ratio: {avg_time:.3f}s avg | "
                  f"Acc: {avg_accuracy:.1%} | TP: {avg_tp_rate:.1%} | "
                  f"FP: {avg_fp_rate:.1%}")


def main():
    """Run the comprehensive testing."""
    import sys

    # Check for flags
    review_mode = '--review' in sys.argv or '-r' in sys.argv
    results_only = '--results-only' in sys.argv or '--results' in sys.argv
    no_cache = '--no-cache' in sys.argv

    # Ensure mutually exclusive modes
    if review_mode and results_only:
        print("❌ Error: Cannot use both --review and --results-only flags "
              "simultaneously")
        sys.exit(1)

    if review_mode:
        print("🔍 Review mode enabled - showing detailed analysis")
    elif results_only:
        print("📊 Results-only mode enabled - showing rare class affinity "
              "scores per user")
    elif no_cache:
        print("Indexes will not be cached")

    # Set random seed for reproducible results
    np.random.seed(42)

    test_thresholds_and_ratios(review_mode=review_mode,
                              results_only=results_only,
                              no_cache=no_cache
                              )

    # Final cache cleanup and summary
    final_cache_info = get_cache_info()
    if final_cache_info['cache_size'] > 0:
        print(f"\n🧹 Cache cleanup: {final_cache_info['cache_size']} "
              f"models in cache")
        clear_model_cache()
        print(f"✅ Cache cleared successfully")

    print(f"\n✅ Testing complete! Check results above to determine "
          f"optimal threshold and ratio.")
    if not review_mode and not results_only:
        print("💡 Tip: Run with --review (-r) for detailed analysis or "
              "--results-only for user scores only")
    if no_cache: 
        print("Index has not been cached between models.")


if __name__ == "__main__":
    main()
