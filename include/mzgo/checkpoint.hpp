// A checkpoint is everything needed to rebuild a network exactly: its shape
// (NetConfig) and every tensor, including batch norm's running statistics,
// which inference depends on but the optimizer never touches. Each tensor is
// stored with its name and size, so loading checks it puts each array back
// where it came from and reports precisely what does not match.
#pragma once
#include <string>
#include "mzgo/network.hpp"
#include "mzgo/serialize.hpp"

namespace mzgo {

// Format: magic "MZGN" (u32 0x4E475A4D, bytes 'M' 'Z' 'G' 'N'), u32 version 3,
// NetConfig as 9 x i32 (boardSize, history, reprBlocks, dynBlocks, filters,
// valueHidden, rewardHead 0/1, unrollSteps, passPlane 0/1; version 2 files
// have the first 8 only and load with passPlane off), u32 tensor count, then per
// tensor: name (str), u64 size, floats. Params in params() order, then
// buffers in buffers() order, one runMean and runVar per batch norm and
// unroll step ("dyn.block0.bn1.runMean.k3"). Version 1 (one set of batch
// norm statistics per layer, no unrollSteps) is refused with a message
// saying the format changed. Written with writeFileAtomically.
void saveNetwork(const MuZeroNetwork& net, const std::string& path);
// Throws std::runtime_error naming path on a missing file, bad magic,
// version, shape, tensor name or size, truncation or trailing data.
MuZeroNetwork loadNetwork(const std::string& path);

// The same format inside a larger file (the trainer's state.bin). readNetwork
// throws std::runtime_error naming `source` on bad contents or truncation; it
// stops after the last tensor (checking for trailing data is the caller's job).
void writeNetwork(BinaryWriter& w, const MuZeroNetwork& net);
MuZeroNetwork readNetwork(BinaryReader& r, const std::string& source);

} // namespace mzgo
