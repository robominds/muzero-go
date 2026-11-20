// A checkpoint is everything needed to rebuild a network exactly: its shape
// (NetConfig) and every tensor, including batch norm's running statistics,
// which inference depends on but the optimizer never touches. Each tensor is
// stored with its name and size, so loading checks it puts each array back
// where it came from and reports precisely what does not match.
#pragma once
#include <string>
#include "mzgo/network.hpp"

namespace mzgo {

// Format: magic "MZGN" (u32 0x4E475A4D, bytes 'M' 'Z' 'G' 'N'), u32 version 1,
// NetConfig as 7 x i32 (boardSize, history, reprBlocks, dynBlocks, filters,
// valueHidden, rewardHead 0/1), u32 tensor count, then per tensor: name
// (str), u64 size, floats. Params in params() order, then buffers in
// buffers() order. Written with writeFileAtomically.
void saveNetwork(const MuZeroNetwork& net, const std::string& path);
// Throws std::runtime_error naming path on a missing file, bad magic,
// version, shape, tensor name or size, truncation or trailing data.
MuZeroNetwork loadNetwork(const std::string& path);

} // namespace mzgo
