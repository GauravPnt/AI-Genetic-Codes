#include <iostream>
#include <queue>
#include <set>
#include <stack>
#include <vector>
using namespace std;

enum { X, Y };

struct State {
  State(int start[2], int capacity[2])
      : bucket{start[0], start[1]}, cap{capacity[0], capacity[1]} {}

  bool operator<(const State &s) const {
    return (bucket[0] < s.bucket[0]) ||
           (bucket[0] == s.bucket[0] && bucket[1] < s.bucket[1]);
  }

  int bucket[2];
  int cap[2];
  State *prev = nullptr;
  int op;
};

string Rules[6] = {"Fill X (Rule 1)",     "Fill Y (Rule 2)",
                   "Empty X (Rule 3)",    "Empty Y (Rule 4)",
                   "Transfer X (Rule 5)", "Transfer Y (Rule 6)"};

State *fill(State *state, bool buck) {
  State *s = new State(*state);
  s->bucket[buck] = s->cap[buck];
  s->op = 0 + int(buck);
  return s;
}

State *empty(State *state, bool buck) {
  State *s = new State(*state);
  s->bucket[buck] = 0;
  s->op = 2 + int(buck);
  return s;
}

State *transfer(State *state, bool buck) {
  State *s = new State(*state);
  int d = min(s->bucket[buck], s->cap[!buck] - s->bucket[!buck]);
  s->bucket[buck] -= d;
  s->bucket[!buck] += d;
  s->op = 4 + int(buck);
  return s;
}

typedef State *(*fptr)(State *, bool);

int main() {
  int start[2];
  cout << "Enter start state of the two buckets:\n";
  cin >> start[0] >> start[1];

  int cap[2];
  cout << "Enter capacity of the two buckets:\n";
  cin >> cap[0] >> cap[1];

  int finish_x;
  cout << "Enter final state of the first bucket:\n";
  cin >> finish_x;

  queue<State *> q;
  set<State> visit;

  State *curr = new State(start, cap);
  q.push(curr);
  visit.insert(*curr);

  fptr prod_rules[3] = {&fill, &empty, &transfer};
  vector<State *> final_state;

  while (!q.empty()) {
    auto top = q.front();
    if (top->bucket[X] == finish_x)
      final_state.push_back(top);
    q.pop();
    for (int b = X; b <= Y; ++b) {
      for (int rule = 0; rule < 3; ++rule) {
        State *s = prod_rules[rule](top, b);
        if (visit.count(*s))
          continue;
        s->prev = top;
        visit.insert(*s);
        q.push(s);
      }
    }
  }

  int method = 1;

  for (auto i : final_state) {
    cout << "--------------------" << '\n';
    cout << '\n';
    cout << "Method " << method++ << '\n' << '\n';
    auto curr = i;
    stack<State *> s;
    while (curr) {
      s.push(curr);
      curr = curr->prev;
    }

    while (!s.empty()) {
      auto top = s.top();
      if (top->bucket[0] != start[0] || top->bucket[1] != start[1])
        cout << "Operation: " << Rules[top->op] << '\n';
      cout << "X: " << top->bucket[X] << " Y: " << top->bucket[Y] << '\n';
      cout << '\n';
      s.pop();
    }
  }
}