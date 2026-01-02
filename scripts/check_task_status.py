#!/usr/bin/env python3
"""
Check Google Earth Engine Task Status
"""

import ee

try:
    ee.Initialize(project='ee-akhiyarwaladi')

    tasks = ee.data.getTaskList()

    print("\n" + "="*80)
    print("GOOGLE EARTH ENGINE - ACTIVE TASKS")
    print("="*80)

    active_found = False

    for task in tasks[:5]:  # Check last 5 tasks
        if task['state'] in ['READY', 'RUNNING']:
            active_found = True
            print(f"\nTask: {task['description']}")
            print(f"  Status: {task['state']}")

            # Calculate progress if available
            if 'progress' in task:
                progress = task['progress'] * 100
                print(f"  Progress: {progress:.1f}%")

    if not active_found:
        print("\n✅ No active tasks - check if completed!")
        print("\nRecent completed tasks:")

        for task in tasks[:3]:
            if task['state'] == 'COMPLETED':
                print(f"\n  ✅ {task['description']}: COMPLETED")

    print("\n" + "="*80)
    print("\nMonitor at: https://code.earthengine.google.com/tasks")
    print("="*80)

except Exception as e:
    print(f"Error: {e}")
    print("Run: earthengine authenticate")
