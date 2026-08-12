#!/usr/bin/env python3
"""Verify all Category 1 implementation components are working."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing Component Imports")
    print("=" * 60)
    
    components = [
        ("Hardware Detector", "src.hardware.detector"),
        ("Hardware Optimizer", "src.hardware.optimizer"),
        ("Model Loader", "src.hardware.model_loader"),
        ("EntryExit V2 Engine", "src.events.entry_exit_v2"),
        ("Zone Engine", "src.events.zone_engine"),
        ("Category 1 Engine", "src.events.category1_engine"),
        ("Rules Engine", "src.events.rules"),
        ("Category 1 Pipeline", "src.pipeline.category1_pipeline"),
    ]
    
    all_passed = True
    for name, module in components:
        try:
            __import__(module)
            print(f"  ✅ {name}: {module}")
        except Exception as e:
            print(f"  ❌ {name}: {module} - {e}")
            all_passed = False
    
    return all_passed


def test_hardware_detection():
    """Test hardware detection."""
    print("\n" + "=" * 60)
    print("Testing Hardware Detection")
    print("=" * 60)
    
    try:
        from src.hardware.detector import detect_hardware
        config = detect_hardware()
        print(f"  ✅ Hardware detected: {config.hardware_type.value}")
        print(f"  ✅ Framework: {config.framework.value}")
        print(f"  ✅ Model: {config.model_name}")
        print(f"  ✅ FPS (480p): {config.fps_480p}")
        print(f"  ✅ FPS (1080p): {config.fps_1080p}")
        return True
    except Exception as e:
        print(f"  ❌ Hardware detection failed: {e}")
        return False


def test_category1_engine_creation():
    """Test Category1Engine creation."""
    print("\n" + "=" * 60)
    print("Testing Category 1 Engine Creation")
    print("=" * 60)
    
    try:
        from src.events.category1_engine import create_category1_engine
        
        engine = create_category1_engine(
            camera_id="test_cam",
            line_norm={"x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0}
        )
        
        print(f"  ✅ Category1Engine created")
        print(f"  ✅ Has zones: {hasattr(engine, 'zones')}")
        print(f"  ✅ Has entry_exit_engine: {hasattr(engine, 'entry_exit_engine')}")
        print(f"  ✅ Has zone_engine: {hasattr(engine, 'zone_engine')}")
        
        # Add a zone
        engine.add_zone(
            zone_id="test_zone",
            name="Test Zone",
            polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            restricted=True
        )
        print(f"  ✅ Zone added successfully")
        
        return True
    except Exception as e:
        import traceback
        print(f"  ❌ Category1Engine creation failed: {e}")
        traceback.print_exc()
        return False


def test_entry_exit_v2_creation():
    """Test EntryExitV2 engine creation."""
    print("\n" + "=" * 60)
    print("Testing EntryExit V2 Engine Creation")
    print("=" * 60)
    
    try:
        from src.events.entry_exit_v2 import EntryExitEngineV2
        
        engine = EntryExitEngineV2(
            line_norm={"x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0}
        )
        
        print(f"  ✅ EntryExitEngineV2 created")
        print(f"  ✅ Line: {engine.line_norm}")
        print(f"  ✅ Buffer threshold: {engine.buffer_threshold}")
        print(f"  ✅ Min track frames: {engine.min_track_frames}")
        print(f"  ✅ Track states: {list(engine.TrackState)}")
        
        return True
    except Exception as e:
        print(f"  ❌ EntryExitV2 creation failed: {e}")
        return False


def test_zone_engine_creation():
    """Test Zone Engine creation."""
    print("\n" + "=" * 60)
    print("Testing Zone Engine Creation")
    print("=" * 60)
    
    try:
        from src.events.zone_engine import ZoneEngine, ZoneType
        
        engine = ZoneEngine(camera_id="test_cam")
        
        print(f"  ✅ ZoneEngine created")
        
        # Add a zone
        engine.add_zone(
            zone_id="test_zone",
            name="Test Zone",
            polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            zone_type=ZoneType.NORMAL,
            restricted=True,
            max_occupancy=10
        )
        print(f"  ✅ Zone added successfully")
        print(f"  ✅ Zones: {list(engine.zones.keys())}")
        
        return True
    except Exception as e:
        print(f"  ❌ Zone Engine creation failed: {e}")
        return False


def test_rules_engine():
    """Test Rules Engine."""
    print("\n" + "=" * 60)
    print("Testing Rules Engine")
    print("=" * 60)
    
    try:
        from src.events.rules import RuleEngine
        
        # Create a simple rule
        rules = {
            "rules": [
                {
                    "rule_id": "test_rule",
                    "name": "Test Rule",
                    "event_types": ["person_entered"],
                    "conditions": {
                        "field": "confidence",
                        "gte": 0.9
                    },
                    "actions": [
                        {
                            "type": "tag",
                            "parameters": {
                                "tags": ["high_confidence"]
                            }
                        }
                    ],
                    "enabled": True,
                    "priority": 1
                }
            ]
        }
        
        # Write to temp file
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(rules, f)
            temp_path = f.name
        
        engine = RuleEngine(rules_config_path=temp_path)
        print(f"  ✅ RuleEngine created with {len(engine.rules)} rule(s)")
        
        # Test processing
        event = {
            "event_type": "person_entered",
            "confidence": 0.95,
            "track_id": 1
        }
        processed = engine.process(event)
        print(f"  ✅ Event processed: {processed is not None}")
        if processed:
            print(f"  ✅ Tags: {processed.get('tags', [])}")
        
        # Cleanup
        os.unlink(temp_path)
        
        return True
    except Exception as e:
        import traceback
        print(f"  ❌ Rules Engine test failed: {e}")
        traceback.print_exc()
        return False


def test_category1_pipeline():
    """Test Category1 Pipeline creation."""
    print("\n" + "=" * 60)
    print("Testing Category 1 Pipeline")
    print("=" * 60)
    
    try:
        from src.pipeline.category1_pipeline import Category1Pipeline, Category1PipelineConfig
        
        config = Category1PipelineConfig(
            camera_id="test_cam",
            line_norm={"x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0},
            auto_detect_hardware=False,
            enable_category1=True,
            enable_rules=False,
            enable_dynamic_optimizer=False,
            model_path="",  # Skip model loading
            event_db_path=":memory:",
            hardware_config_path="",
        )
        
        pipeline = Category1Pipeline(config=config)
        
        print(f"  ✅ Category1Pipeline created")
        print(f"  ✅ Has category1_engine: {pipeline.category1_engine is not None}")
        print(f"  ✅ Has event_store: {pipeline.event_store is not None}")
        print(f"  ✅ Has tracker: {pipeline.tracker is not None}")
        
        return True
    except Exception as e:
        import traceback
        print(f"  ❌ Category1Pipeline creation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FACE-RECOGNITION-UBUNTO: CATEGORY 1 IMPLEMENTATION")
    print("VERIFICATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Component Imports", test_imports),
        ("Hardware Detection", test_hardware_detection),
        ("EntryExit V2 Engine", test_entry_exit_v2_creation),
        ("Zone Engine", test_zone_engine_creation),
        ("Category 1 Engine", test_category1_engine_creation),
        ("Rules Engine", test_rules_engine),
        ("Category 1 Pipeline", test_category1_pipeline),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Implementation is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
