# MuZero Go

A from-scratch C++17 implementation of MuZero for the game of Go: a learned
model (representation, dynamics and prediction networks) that plans with
Monte Carlo tree search without being given the rules. It is a sibling of
alphazero-go and muzero-tictactoe.

## Build

```
# portable, no dependencies
cmake -S . -B build-naive -DCMAKE_BUILD_TYPE=Release
cmake --build build-naive -j && ctest --test-dir build-naive --output-on-failure

# with BLAS (Accelerate on macOS, OpenBLAS elsewhere)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMZ_USE_BLAS=ON
cmake --build build -j && ctest --test-dir build --output-on-failure
```

## Authorship

Mark Castelluccio, designed and implemented with Claude Code (AI-assisted).
