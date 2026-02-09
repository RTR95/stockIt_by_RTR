
try:
    from src.analysis.signal_generator import SignalResult
    print("SignalResult imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
