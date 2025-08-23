# HyperConformal Quality Gates Report\nGenerated at: 2025-08-23 04:10:56\n\n## Quality Gate: LINTING\nStatus: ✅ PASS\n\n## Quality Gate: TYPE CHECKING\nStatus: ✅ PASS\n\n## Quality Gate: SECURITY\nStatus: ❌ FAIL\nDuration: 0.05s\n### Output:\n```\nSecurity issues found: 2
  - hyperconformal/monitoring.py: Contains eval()
  - hyperconformal/quantum_conformal_research.py: Contains eval()
\n```\n\n## Quality Gate: UNIT TESTS\nStatus: ❌ FAIL\nDuration: 9.73s\n### Output:\n```\n============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-7.4.4, pluggy-1.4.0
rootdir: /root/repo
configfile: pytest.ini
collected 168 items

tests/test_advanced_features.py ........F.......F..FFF
============================================================
HyperConformal Test Suite Complete
============================================================


=================================== FAILURES ===================================
_________________ TestFederatedFeatures.test_federated_client __________________
tests/test_advanced_features.py:194: in test_federated_client
    client.initialize_model(global_state)
hyperconformal/federated.py:196: in initialize_model
    self.local_encoder.load_state_dict(global_encoder_state)
/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py:2624: in load_state_dict
    raise RuntimeError(
E   RuntimeError: Error(s) in loading state_dict for RandomProjection:
E   	Missing k...\n```\n### Errors:\n```\nNeuromorphic backends not available. Install nengo and nengo_loihi for full functionality.
\n```\n\n## Quality Gate: PERFORMANCE\nStatus: ✅ PASS\nDuration: 4.03s\n### Output:\n```\nRunning performance tests...
Training time: 0.063s
Prediction time: 0.031s
Throughput: 32439.0 samples/sec
Performance tests completed
\n```\n\n## Quality Gate: MEMORY\nStatus: ✅ PASS\nDuration: 4.55s\n### Output:\n```\nRunning memory tests...
Initial memory: 0.0MB
After training: 0.0MB
After prediction: 0.0MB
After cleanup: 0.0MB
Memory test passed
\n```\n\n## Overall Result: ❌ SOME GATES FAILED\n