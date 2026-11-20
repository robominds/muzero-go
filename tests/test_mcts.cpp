#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>
#include "fake_model.hpp"
#include "mzgo/game.hpp"
#include "mzgo/mcts.hpp"

using namespace mzgo;

// 3x3 board with 2 filters: 10 actions (pass = 9), hidden size 18.
static const ModelDims kDims{3, 2};
static const int kActions = 10, kPass = 9, kHs = 18;

static bool near(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) < eps; }

// Runs a whole search from given root outputs (instead of a real Game, so a
// test can pick the legal set), answering each leaf with `session`. When
// `nonRootLeaves` is given, it collects the actions of leaves whose parent is
// not the root (depth >= 2).
static SearchResult searchFrom(Search& s, ModelSession& session, const std::vector<float>& rootLogits,
                               const std::vector<int>& legal, bool previousMoveWasPass, Rng& rng,
                               std::vector<int>* nonRootLeaves = nullptr) {
    const std::vector<float> rootHidden(kHs, 0.5f);
    s.beginRoot(rootHidden.data(), rootLogits.data(), 0.0f, legal, previousMoveWasPass, rng);
    ModelOutput out;
    while (!s.finished()) {
        LeafRequest req = s.selectLeaf();
        if (nonRootLeaves && !std::equal(rootHidden.begin(), rootHidden.end(), req.parentHidden))
            nonRootLeaves->push_back(req.action);
        session.recurrentInference(req.parentHidden, &req.action, &req.depth, 1, out);
        s.expandLeaf(out.hidden.data(), out.logits.data(), out.values[0], out.rewards[0]);
    }
    return s.result();
}

static void test_minmax() {
    MinMaxStats m;                       // known bounds [-1, 1]
    assert(near(m.normalize(-1.0f), 0.0f));
    assert(near(m.normalize(1.0f), 1.0f));
    assert(near(m.normalize(0.0f), 0.5f));
    m.update(0.3f);                      // inside the bounds: no change
    assert(near(m.normalize(0.0f), 0.5f));
    m.update(3.0f);                      // widens to [-1, 3]
    assert(near(m.normalize(3.0f), 1.0f));
    assert(near(m.normalize(1.0f), 0.5f));
    assert(near(m.normalize(-1.0f), 0.0f));
    MinMaxStats flat(0.0f, 0.0f);        // empty range: q unchanged
    assert(near(flat.normalize(0.7f), 0.7f));
    std::printf("test_minmax passed\n");
}

static void test_ucb_hand_computed() {
    // Root priors {0.5, 0.3, 0.2} over three legal actions, placed so the
    // largest prior is on the highest action: A = action 6 (0.5),
    // B = action 4 (0.3), C = action 1 (0.2). The other logits are so low
    // they would be ~0 anyway, and are masked. Leaf values are fed by hand.
    // c1 = 1.25, c2 = 19652, pbC(N) = c1 + ln((N + c2 + 1)/c2).
    // Q is normalized on the known bounds [-1, 1]: Q̄ = (Q + 1)/2.
    //
    // sim 1: N = 0, every score is 0 -> tie, larger prior wins over lower
    //        action: A. Leaf value +1 (child's view) -> QA = -1, Q̄A = 0.
    // sim 2: N = 1, pbC = 1.25 + ln(19654/19652) = 1.2501018
    //        A: 0 + 0.5*1/2*1.2501018 = 0.3125
    //        B: 0 + 0.3*1/1*1.2501018 = 0.3750   <- selected
    //        C: 0 + 0.2*1/1*1.2501018 = 0.2500
    //        Leaf value +0.2 -> QB = -0.2, Q̄B = 0.4.
    // sim 3: N = 2, pbC = 1.25 + ln(19655/19652) = 1.2501526, sqrt 2 = 1.4142136
    //        A: 0   + 0.5*1.4142136/2*1.2501526 = 0.4420
    //        B: 0.4 + 0.3*1.4142136/2*1.2501526 = 0.6652   <- selected
    //        C: 0   + 0.2*1.4142136/1*1.2501526 = 0.3536
    //        B is expanded, so this descends to its child 0 (uniform prior,
    //        tie -> lowest action). Leaf value -0.8 at depth 2: B gets +0.8,
    //        B mean = (0.2 + 0.8)/2 = 0.5 -> QB = -0.5, Q̄B = 0.25.
    // sim 4: N = 3, pbC = 1.25 + ln(19656/19652) = 1.2502035, sqrt 3 = 1.7320508
    //        A: 0    + 0.5*1.7320508/2*1.2502035 = 0.5414   <- selected
    //        B: 0.25 + 0.3*1.7320508/3*1.2502035 = 0.4665
    //        C: 0    + 0.2*1.7320508/1*1.2502035 = 0.4331
    //        The best-valued edge (B) and the unvisited one (C) both lose
    //        to the exploration term of A. Leaf value 0 at depth 2.
    // Root value: (-1 - 0.2 - 0.8 + 0)/4 = -0.5. Depths 1, 1, 2, 2.
    const int A = 6, B = 4, C = 1;
    MctsConfig cfg;
    cfg.simulations = 4;
    Search s(cfg, kDims);
    std::vector<float> logits(kActions, -30.0f);
    logits[A] = std::log(0.5f);
    logits[B] = std::log(0.3f);
    logits[C] = std::log(0.2f);
    const std::vector<float> rootHidden(kHs, 0.0f), zeros(kActions, 0.0f);
    std::vector<std::vector<float>> h;
    for (int k = 1; k <= 4; ++k) h.emplace_back(kHs, 0.1f * k);
    Rng rng(1);
    s.beginRoot(rootHidden.data(), logits.data(), 0.0f, {C, B, A}, false, rng);
    std::vector<RootEdge> e = s.rootEdges();                  // ascending action: C, B, A
    assert(e.size() == 3 && e[0].action == C && e[1].action == B && e[2].action == A);
    assert(near(e[0].prior, 0.2f) && near(e[1].prior, 0.3f) && near(e[2].prior, 0.5f));

    LeafRequest r = s.selectLeaf();                           // sim 1
    assert(r.action == A && r.parentHidden[0] == 0.0f && r.depth == 1);
    s.expandLeaf(h[0].data(), zeros.data(), 1.0f, 0.0f);
    assert(near(s.rootEdges()[2].q, -1.0f));

    r = s.selectLeaf();                                       // sim 2
    assert(r.action == B && r.parentHidden[0] == 0.0f && r.depth == 1);
    s.expandLeaf(h[1].data(), zeros.data(), 0.2f, 0.0f);

    r = s.selectLeaf();                                       // sim 3: under B
    assert(r.action == 0 && r.parentHidden[0] == h[1][0] && r.depth == 2);
    s.expandLeaf(h[2].data(), zeros.data(), -0.8f, 0.0f);
    e = s.rootEdges();
    assert(e[1].visits == 2 && near(e[1].meanValue, 0.5f) && near(e[1].q, -0.5f));

    r = s.selectLeaf();                                       // sim 4: under A
    assert(r.action == 0 && r.parentHidden[0] == h[0][0] && r.depth == 2);
    s.expandLeaf(h[3].data(), zeros.data(), 0.0f, 0.0f);
    assert(s.finished());

    SearchResult res = s.result();
    assert(res.visits[A] == 2 && res.visits[B] == 2 && res.visits[C] == 0);
    assert(near(res.policy[A], 0.5f) && near(res.policy[B], 0.5f) && res.policy[C] == 0.0f);
    assert(near(res.rootValue, -0.5f));
    assert(res.maxDepth == 2 && near(res.meanDepth, 1.5f));
    std::printf("test_ucb_hand_computed passed\n");
}

static void test_backup_signs() {
    // Every leaf says +0.8 for its own side to move.
    FakeModel model(kDims, 1, nullptr, [](const float*, int, FakeOutput& o) { o.value = 0.8f; });
    auto session = model.newSession(1);
    const std::vector<float> logits(kActions, 0.0f);
    std::vector<int> all;
    for (int a = 0; a < kActions; ++a) all.push_back(a);

    MctsConfig cfg;
    cfg.simulations = 1;
    Search s(cfg, kDims);
    Rng rng(1);
    SearchResult r = searchFrom(s, *session, logits, all, false, rng);
    std::vector<RootEdge> e = s.rootEdges();
    assert(e[0].visits == 1);                 // uniform priors, tie -> action 0
    assert(near(e[0].meanValue, 0.8f));       // child's own view
    assert(near(e[0].q, -0.8f));              // good for the child = bad for the root
    assert(near(r.rootValue, -0.8f));

    // A second simulation goes one ply deeper under action 0 (Q̄0 = 0.1 plus
    // its exploration term 0.0625 beats 0.125 of the unvisited edges). Its
    // +0.8 is -0.8 for action 0's node and +0.8 for the root.
    cfg.simulations = 2;
    Search s2(cfg, kDims);
    r = searchFrom(s2, *session, logits, all, false, rng);
    e = s2.rootEdges();
    assert(e[0].visits == 2 && near(e[0].meanValue, 0.0f));
    assert(near(r.rootValue, 0.0f) && r.maxDepth == 2);
    std::printf("test_backup_signs passed\n");
}

// Rewards on the edges: every leaf has reward +0.6 on its incoming edge and
// value -0.5 for its own side to move, so an edge into a fresh leaf is worth
// Q = r - v_child = 0.6 + 0.5 = 1.1 to the parent: outside the [-1, 1] a
// board game's values live in, which MinMaxStats has to widen for.
static void test_rewards_and_minmax_widening() {
    FakeModel model(kDims, 1, nullptr, [](const float*, int, FakeOutput& o) {
        o.reward = 0.6f;
        o.value = -0.5f;
    });
    auto session = model.newSession(1);
    const std::vector<float> logits(kActions, 0.0f);

    // One legal root action (4), so the second simulation goes one deeper.
    // sim 1: leaf L1 = child 4 of the root, v = -0.5, r = 0.6.
    //        L1: N 1, sum -0.5. Root: += r(L1) - (-0.5) = 1.1.
    // sim 2: root -> L1 -> L2 (L1's children have uniform priors: action 0).
    //        L2: N 1, sum -0.5.
    //        L1: += r(L2) - v(L2) = 0.6 + 0.5 = 1.1 -> N 2, sum 0.6, mean 0.3.
    //        Root: += r(L1) - 1.1 = -0.5 -> N 2, sum 0.6, mean 0.3.
    //        Root edge Q = r(L1) - mean(L1) = 0.6 - 0.3 = 0.3.
    // The Q values seen: 1.1 (sim 1's edge, and L2's edge), 0.3: hi = 1.1.
    MctsConfig cfg;
    cfg.simulations = 1;
    Search one(cfg, kDims);
    Rng rng(1);
    SearchResult r = searchFrom(one, *session, logits, {4}, false, rng);
    std::vector<RootEdge> e = one.rootEdges();
    assert(e.size() == 1 && e[0].visits == 1);
    assert(near(e[0].meanValue, -0.5f) && near(e[0].q, 1.1f));   // Q = r - v_child
    assert(near(r.rootValue, 1.1f));

    cfg.simulations = 2;
    Search two(cfg, kDims);
    r = searchFrom(two, *session, logits, {4}, false, rng);
    e = two.rootEdges();
    assert(e[0].visits == 2 && near(e[0].meanValue, 0.3f) && near(e[0].q, 0.3f));
    assert(near(r.rootValue, 0.3f) && r.maxDepth == 2);
    assert(near(two.minMaxStats().hi(), 1.1f) && two.minMaxStats().lo() == -1.0f);

    // A full search over every action: the range widened above 1, and every
    // visited root edge's normalized Q stays inside [0, 1].
    std::vector<int> all;
    for (int a = 0; a < kActions; ++a) all.push_back(a);
    cfg.simulations = 60;
    Search wide(cfg, kDims);
    searchFrom(wide, *session, logits, all, false, rng);
    const MinMaxStats& mm = wide.minMaxStats();
    assert(mm.hi() > 1.0f);
    int visited = 0;
    for (const RootEdge& edge : wide.rootEdges()) {
        if (edge.visits == 0) continue;
        ++visited;
        const float q = mm.normalize(edge.q);
        assert(q >= 0.0f && q <= 1.0f);
    }
    assert(visited >= 2);
    std::printf("test_rewards_and_minmax_widening passed\n");
}

static void test_root_masking() {
    // The model loves action 1, which is illegal at the root.
    FakeModel model(kDims, 1, nullptr, [](const float*, int, FakeOutput& o) { o.logits[1] = 5.0f; });
    auto session = model.newSession(1);
    std::vector<float> logits(kActions, 0.0f);
    logits[1] = 5.0f;
    const std::vector<int> legal = {0, 5, kPass};

    MctsConfig cfg;
    cfg.simulations = 50;
    Search s(cfg, kDims);
    Rng rng(1);
    std::vector<int> deep;
    SearchResult r = searchFrom(s, *session, logits, legal, false, rng, &deep);
    int total = 0;
    for (int a = 0; a < kActions; ++a) {
        if (a != 0 && a != 5 && a != kPass) assert(r.visits[a] == 0 && r.policy[a] == 0.0f);
        total += r.visits[a];
    }
    assert(total == 50 && s.rootEdges().size() == 3);
    // Below the root nothing is masked: the favourite is chosen there.
    int chosenDeep = 0;
    for (int a : deep) chosenDeep += (a == 1);
    assert(chosenDeep > 0);
    std::printf("test_root_masking passed\n");
}

static void test_single_legal_root() {
    FakeModel model(kDims);
    auto session = model.newSession(1);
    const std::vector<float> logits(kActions, 0.0f);
    MctsConfig cfg;
    cfg.simulations = 20;
    Search s(cfg, kDims);
    Rng rng(1);
    SearchResult r = searchFrom(s, *session, logits, {kPass}, false, rng);
    for (int a = 0; a < kPass; ++a) assert(r.visits[a] == 0 && r.policy[a] == 0.0f);
    assert(r.visits[kPass] == 20 && r.policy[kPass] == 1.0f);
    assert(s.selectMove(r, false, rng) == kPass && s.selectMove(r, true, rng) == kPass);
    std::printf("test_single_legal_root passed\n");
}

static void test_noise() {
    std::vector<float> logits(kActions, 0.0f);
    logits[3] = -1000.0f;                    // softmax prior underflows to exactly 0
    const std::vector<int> legal = {0, 3, 5, kPass};
    const std::vector<float> hidden(kHs, 0.0f);
    MctsConfig cfg;
    Rng rng(3);

    Search plain(cfg, kDims);
    plain.beginRoot(hidden.data(), logits.data(), 0.0f, legal, false, rng);
    assert(plain.rootEdges()[1].action == 3 && plain.rootEdges()[1].prior == 0.0f);

    cfg.addNoise = true;
    Search noisy(cfg, kDims);
    noisy.beginRoot(hidden.data(), logits.data(), 0.0f, legal, false, rng);
    float sum = 0.0f;
    for (const RootEdge& e : noisy.rootEdges()) {
        assert(e.prior > 0.0f);
        sum += e.prior;
    }
    assert(near(sum, 1.0f));
    std::printf("test_noise passed\n");
}

static void test_past_end_fraction() {
    // Pass is the model's clear favourite, at the root and below.
    auto passy = [](FakeOutput& o) { o.logits[kPass] = 10.0f; };
    FakeModel model(kDims, 2, [&](const float*, FakeOutput& o) { passy(o); },
                    [&](const float*, int, FakeOutput& o) { passy(o); });
    auto session = model.newSession(1);
    MctsConfig cfg;
    cfg.simulations = 50;

    Game afterPass(3, 0.5f, defaultMaxMoves(3));
    afterPass.play(kPass);                   // a further pass would end the real game
    Search s(cfg, kDims);
    Rng rng(1);
    SearchResult r = runSearch(s, *session, afterPass, model.history(), rng);
    assert(r.pastEndFraction > 0.5f);

    // The real previous move counts: one simulation choosing pass is past the
    // end after a pass, and not at the start of a game.
    cfg.simulations = 1;
    Search one(cfg, kDims);
    r = runSearch(one, *session, afterPass, model.history(), rng);
    assert(r.visits[kPass] == 1 && r.pastEndFraction == 1.0f && r.maxDepth == 1);
    Game fresh(3, 0.5f, defaultMaxMoves(3));
    r = runSearch(one, *session, fresh, model.history(), rng);
    assert(r.visits[kPass] == 1 && r.pastEndFraction == 0.0f);
    std::printf("test_past_end_fraction passed\n");
}

static void test_determinism() {
    // Values and priors that vary with the hidden state, so the tree is uneven.
    auto shape = [](FakeOutput& o) {
        o.value = 2.0f * o.hidden[0] - 1.0f;
        for (int a = 0; a < kActions; ++a) o.logits[a] = 3.0f * o.hidden[a % kHs];
    };
    FakeModel model(kDims, 2, [&](const float*, FakeOutput& o) { shape(o); },
                    [&](const float*, int, FakeOutput& o) { shape(o); });
    auto session = model.newSession(1);
    Game g(3, 0.5f, defaultMaxMoves(3));
    g.play(4);
    g.play(0);
    MctsConfig cfg;
    cfg.simulations = 64;
    cfg.addNoise = true;

    Search a(cfg, kDims), b(cfg, kDims);
    Rng ra(7), rb(7);
    SearchResult x = runSearch(a, *session, g, model.history(), ra);
    SearchResult y = runSearch(b, *session, g, model.history(), rb);
    assert(x.visits == y.visits && x.rootValue == y.rootValue);
    assert(x.maxDepth == y.maxDepth && x.meanDepth == y.meanDepth);
    assert(a.selectMove(x, true, ra) == b.selectMove(y, true, rb));
    assert(x.maxDepth >= 1 && x.meanDepth >= 1.0f && x.meanDepth <= static_cast<float>(x.maxDepth));
    for (int act = 0; act < kActions; ++act)
        if (!g.isLegal(act)) assert(x.visits[act] == 0);

    // The same Search object, reused: beginRoot clears the old tree.
    Rng rc(7);
    SearchResult z = runSearch(a, *session, g, model.history(), rc);
    assert(z.visits == x.visits && z.rootValue == x.rootValue);
    std::printf("test_determinism passed\n");
}

static void test_select_move() {
    MctsConfig cfg;
    Search s(cfg, kDims);
    SearchResult r;
    r.visits.assign(kActions, 0);
    r.visits[2] = 3;
    r.visits[5] = 3;
    r.visits[7] = 1;
    Rng rng(5);
    assert(s.selectMove(r, false, rng) == 2);       // argmax, tie -> lowest index

    r.visits.assign(kActions, 0);
    r.visits[1] = 1;
    r.visits[6] = 3;
    int six = 0;
    const int draws = 4000;
    for (int i = 0; i < draws; ++i) {
        int m = s.selectMove(r, true, rng);
        assert(m == 1 || m == 6);
        six += (m == 6);
    }
    assert(std::fabs(six / static_cast<float>(draws) - 0.75f) < 0.03f);
    std::printf("test_select_move passed\n");
}

int main() {
    test_minmax();
    test_ucb_hand_computed();
    test_backup_signs();
    test_rewards_and_minmax_widening();
    test_root_masking();
    test_single_legal_root();
    test_noise();
    test_past_end_fraction();
    test_determinism();
    test_select_move();
    std::printf("test_mcts: all tests passed\n");
    return 0;
}
