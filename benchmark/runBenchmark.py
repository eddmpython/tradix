"""
tradeBT 성능 벤치마크

v1 (Numba) vs v2 (Pure NumPy) 비교
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import numpy as np


def benchmark():
    np.random.seed(42)
    n = 2500
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)

    print(f"데이터 크기: {n}개 (약 10년 일봉)")
    print("=" * 60)

    try:
        from tradix import v1
        hasV1 = True
        print("v1 (Numba): 로드 완료")
    except ImportError as e:
        hasV1 = False
        print(f"v1 (Numba): 로드 실패 - {e}")

    try:
        from tradix import v2
        hasV2 = True
        print("v2 (NumPy): 로드 완료")
    except ImportError as e:
        hasV2 = False
        print(f"v2 (NumPy): 로드 실패 - {e}")

    print("=" * 60)

    iterations = 100

    tests = [
        ("vsma(20)", lambda m: m.vsma(close, 20)),
        ("vema(20)", lambda m: m.vema(close, 20)),
        ("vrsi(14)", lambda m: m.vrsi(close, 14)),
        ("vroc(12)", lambda m: m.vroc(close, 12)),
        ("vmomentum(10)", lambda m: m.vmomentum(close, 10)),
    ]

    results = []

    for name, func in tests:
        v1Time = None
        v2Time = None

        if hasV1:
            for _ in range(10):
                func(v1)
            start = time.perf_counter()
            for _ in range(iterations):
                func(v1)
            v1Time = (time.perf_counter() - start) / iterations * 1000

        if hasV2:
            for _ in range(10):
                func(v2)
            start = time.perf_counter()
            for _ in range(iterations):
                func(v2)
            v2Time = (time.perf_counter() - start) / iterations * 1000

        results.append((name, v1Time, v2Time))

    print(f"\n{'함수':<20} {'v1(Numba)':<15} {'v2(NumPy)':<15} {'결과':<20}")
    print("-" * 70)

    for name, v1Time, v2Time in results:
        v1Str = f"{v1Time:.4f}ms" if v1Time else "N/A"
        v2Str = f"{v2Time:.4f}ms" if v2Time else "N/A"

        if v1Time and v2Time:
            if v1Time < v2Time:
                diffStr = f"v1이 {v2Time/v1Time:.1f}x 빠름"
            else:
                diffStr = f"v2가 {v1Time/v2Time:.1f}x 빠름"
        else:
            diffStr = "-"

        print(f"{name:<20} {v1Str:<15} {v2Str:<15} {diffStr:<20}")

    print("\n" + "=" * 60)

    if hasV1 and hasV2:
        v1Total = sum(t for _, t, _ in results if t)
        v2Total = sum(t for _, _, t in results if t)
        print(f"v1 총합: {v1Total:.3f}ms | v2 총합: {v2Total:.3f}ms")

        if v1Total < v2Total:
            print(f"→ v1 (Numba)이 {v2Total/v1Total:.1f}x 빠름")
        else:
            print(f"→ v2 (NumPy)가 {v1Total/v2Total:.1f}x 빠름")


if __name__ == "__main__":
    benchmark()
