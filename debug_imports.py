
try:
    from src.analysis.signal_generator import SignalGenerator
    print("Import successful")
except ImportError as e:
    import traceback
    traceback.print_exc()
except Exception as e:
    import traceback
    traceback.print_exc()
