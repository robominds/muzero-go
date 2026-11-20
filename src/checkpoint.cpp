#include "mzgo/checkpoint.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "mzgo/board.hpp"
#include "mzgo/serialize.hpp"

namespace mzgo {

namespace {

constexpr uint32_t kMagic = 0x4E475A4D;   // bytes 'M' 'Z' 'G' 'N' when written little-endian
// Version 2: batch norm running statistics per unroll step (network.hpp),
// so the header carries unrollSteps and the buffers are named per set.
constexpr uint32_t kVersion = 2;
// Shape limits for a header to be believed: they keep a corrupt file from
// asking for an absurd allocation.
constexpr int kMaxHistory = 16, kMaxBlocks = 64, kMaxFilters = 1024, kMaxValueHidden = 1024, kMaxUnroll = 64;

// A problem found in the file's contents (as opposed to running out of data).
struct FormatError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Tensor {
    std::string name;
    float* data;
    size_t size;
};

std::vector<Tensor> tensorsOf(MuZeroNetwork& net) {
    std::vector<Tensor> out;
    for (const ParamView& p : net.params()) out.push_back({p.name, p.value, p.size});
    for (const BufferView& b : net.buffers()) out.push_back({b.name, b.value, b.size});
    return out;
}

NetConfig readHeader(BinaryReader& r) {
    if (r.u32() != kMagic) throw FormatError("not an MZGN checkpoint");
    const uint32_t version = r.u32();
    if (version == 1)
        throw FormatError("checkpoint format version 1 is no longer supported: the format changed to version 2, "
                          "which keeps batch norm statistics per unroll step; networks and runs saved before "
                          "that change cannot be loaded or resumed");
    if (version != kVersion) throw FormatError("unsupported version " + std::to_string(version));
    NetConfig cfg;
    cfg.boardSize = r.i32();
    cfg.history = r.i32();
    cfg.reprBlocks = r.i32();
    cfg.dynBlocks = r.i32();
    cfg.filters = r.i32();
    cfg.valueHidden = r.i32();
    const int32_t reward = r.i32();
    if (reward != 0 && reward != 1) throw FormatError("invalid reward head flag in header");
    cfg.rewardHead = reward == 1;
    cfg.unrollSteps = r.i32();
    if (cfg.boardSize < kMinBoardSize || cfg.boardSize > kMaxBoardSize || cfg.history < 1 ||
        cfg.history > kMaxHistory || cfg.reprBlocks < 0 || cfg.reprBlocks > kMaxBlocks || cfg.dynBlocks < 0 ||
        cfg.dynBlocks > kMaxBlocks || cfg.filters < 1 || cfg.filters > kMaxFilters || cfg.valueHidden < 1 ||
        cfg.valueHidden > kMaxValueHidden || cfg.unrollSteps < 1 || cfg.unrollSteps > kMaxUnroll) {
        throw FormatError("invalid network shape in header");
    }
    return cfg;
}

} // namespace

void writeNetwork(BinaryWriter& w, const MuZeroNetwork& net) {
    const NetConfig& cfg = net.config();
    w.u32(kMagic);
    w.u32(kVersion);
    for (int v : {cfg.boardSize, cfg.history, cfg.reprBlocks, cfg.dynBlocks, cfg.filters, cfg.valueHidden})
        w.i32(v);
    w.i32(cfg.rewardHead ? 1 : 0);
    w.i32(cfg.unrollSteps);
    // params() hands out writable views; here they are only read.
    const std::vector<Tensor> tensors = tensorsOf(const_cast<MuZeroNetwork&>(net));
    w.u32(static_cast<uint32_t>(tensors.size()));
    for (const Tensor& t : tensors) {
        w.str(t.name);
        w.u64(t.size);
        w.floats(t.data, t.size);
    }
}

MuZeroNetwork readNetwork(BinaryReader& r, const std::string& source) {
    try {
        MuZeroNetwork net(readHeader(r), 0);   // the seed is irrelevant: every tensor is overwritten
        const std::vector<Tensor> tensors = tensorsOf(net);
        const uint32_t count = r.u32();
        if (count != tensors.size())
            throw FormatError("has " + std::to_string(count) + " tensors but the network needs " +
                              std::to_string(tensors.size()));
        for (size_t i = 0; i < tensors.size(); ++i) {
            const std::string name = r.str();
            if (name != tensors[i].name)
                throw FormatError("tensor " + std::to_string(i) + " is '" + name + "' but expected '" +
                                  tensors[i].name + "'");
            const uint64_t size = r.u64();
            if (size != tensors[i].size)
                throw FormatError("tensor '" + name + "' has " + std::to_string(size) + " values but expected " +
                                  std::to_string(tensors[i].size));
            r.floats(tensors[i].data, tensors[i].size);
        }
        return net;
    } catch (const FormatError& e) {
        throw std::runtime_error(source + ": " + e.what());
    } catch (const std::runtime_error&) {
        // BinaryReader's own error: the data ended early.
        throw std::runtime_error(source + ": truncated");
    }
}

void saveNetwork(const MuZeroNetwork& net, const std::string& path) {
    std::ostringstream buffer(std::ios::binary);
    BinaryWriter w(buffer);
    writeNetwork(w, net);
    writeFileAtomically(path, buffer.str());
}

MuZeroNetwork loadNetwork(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open checkpoint " + path);
    BinaryReader r(in);
    MuZeroNetwork net = readNetwork(r, path);
    if (in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error(path + ": unexpected data after the last tensor");
    return net;
}

} // namespace mzgo
