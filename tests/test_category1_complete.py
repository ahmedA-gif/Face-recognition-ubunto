#!/usr/bin/env python3
"""Complete test for Category 1 implementation."""

from src.events.entry_exit_v2 import EntryExitEngineV2
from src.events.store import EventsStore
from src.tracking.bytetrack import Track
import numpy as np


def test_entry_exit_5layer():
    """Test the 5-layer entry/exit validation."""
    print("=" * 60)
    print("Testing EntryExitEngineV2 with 5-Layer Validation")
    print("=" * 60)
    
    # Create engine with relaxed parameters for testing
    engine = EntryExitEngineV2(
        line_norm={'x1': 0.5, 'y1': 0.0, 'x2': 0.5, 'y2': 1.0},
        min_track_frames=1,
        min_deep_frames=1,
        buffer_threshold=20.0,
    )
    
    print(f'✅ EntryExitEngineV2 created')
    print(f'   Line: {engine.line_norm}')
    print(f'   Buffer threshold: {engine.buffer_threshold}')
    
    store = EventsStore(db_path=':memory:')
    
    # Simulate person moving from outside to inside
    print("\nSimulating person crossing from OUTSIDE to INSIDE:")
    
    positions = [
        (300, "Far outside"),
        (400, "Outside"),
        (470, "Approaching"),
        (500, "Buffer"),
        (530, "Crossing"),
        (600, "Inside"),
        (700, "Deep inside"),
    ]
    
    all_events = []
    for x, description in positions:
        track = Track(
            track_id=1,
            xyxy=np.array([x, 300, x+40, 380]),
            conf=0.95,
            hits=5
        )
        events = engine.update([track], (720, 1280, 3), store)
        all_events.extend(events)
        print(f'  Frame (x={x}): {len(events)} events - {description}')
    
    print(f'\n📊 Total events generated: {len(all_events)}')
    for i, event in enumerate(all_events):
        print(f'  Event {i+1}:')
        if hasattr(event, 'direction'):
            print(f'    Direction: {event.direction}')
        if hasattr(event, 'confidence'):
            print(f'    Confidence: {event.confidence:.2f}')
        if hasattr(event, 'track_id'):
            print(f'    Track ID: {event.track_id}')
    
    # Test return crossing (EXIT)
    print("\n" + "=" * 60)
    print("Simulating person crossing from INSIDE to OUTSIDE:")
    
    engine2 = EntryExitEngineV2(
        line_norm={'x1': 0.5, 'y1': 0.0, 'x2': 0.5, 'y2': 1.0},
        min_track_frames=1,
        min_deep_frames=1,
        buffer_threshold=20.0,
    )
    
    positions_exit = [
        (700, "Deep inside"),
        (600, "Inside"),
        (530, "Crossing"),
        (500, "Buffer"),
        (470, "Approaching exit"),
        (400, "Outside"),
        (300, "Far outside"),
    ]
    
    all_events_exit = []
    for x, description in positions_exit:
        track = Track(
            track_id=2,
            xyxy=np.array([x, 300, x+40, 380]),
            conf=0.95,
            hits=5
        )
        events = engine2.update([track], (720, 1280, 3), store)
        all_events_exit.extend(events)
        print(f'  Frame (x={x}): {len(events)} events - {description}')
    
    print(f'\n📊 Total exit events generated: {len(all_events_exit)}')
    for i, event in enumerate(all_events_exit):
        print(f'  Event {i+1}:')
        if hasattr(event, 'direction'):
            print(f'    Direction: {event.direction}')
        if hasattr(event, 'confidence'):
            print(f'    Confidence: {event.confidence:.2f}')
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    entry_events = [e for e in all_events if hasattr(e, 'direction') and e.direction == 'entry']
    exit_events = [e for e in all_events_exit if hasattr(e, 'direction') and e.direction == 'exit']
    print(f'✅ Entry events: {len(entry_events)}')
    print(f'✅ Exit events: {len(exit_events)}')
    
    if len(entry_events) >= 1 and len(exit_events) >= 1:
        print('\n✅ ALL TESTS PASSED!')
        print('   5-Layer validation is working correctly')
        return True
    else:
        print('\n⚠️  WARNING: Expected at least 1 entry and 1 exit event')
        print('   This may be due to the 5-layer validation requiring more frames')
        return False


if __name__ == '__main__':
    success = test_entry_exit_5layer()
    exit(0 if success else 1)
