#include <bits/stdc++.h>
using namespace std;

enum { E, Q };

int q;
int n;

bool check(int x, int y) { return x < n && y < n && x >= 0 && y >= 0; }

bool test(vector<vector<int>> &board) {
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      if (board[row][col] == Q) {
        for (int j = 0; j < n; ++j) {
          if (j != row && board[j][col] == Q)
            return false;
          if (row - col + j != row && check(row - col + j, j) &&
              board[row - col + j][j] == Q)
            return false;
          if (row + col - j != row && check(row + col - j, j) &&
              board[row + col - j][j] == Q)
            return false;
        }
      }
    }
  }
  return true;
}

void print(vector<vector<int>> &board) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (board[i][j] == Q)
        cout << "Q ";
      else
        cout << "* ";
    }
    cout << '\n';
  }
}

int main() {
  cin >> n;
  int cnt = 0;
  vector<vector<int>> board(n, vector<int>(n));
  random_device rd;
  mt19937 g(rd());
  for (int i = 0; i < n; ++i) {
    board[i][0] = Q;
  }
  while (1) {
    for (int i = 0; i < n; ++i) {
      shuffle(board[i].begin(), board[i].end(), g);
    }
    if (test(board)) {
      cout << cnt << '\n';
      print(board);
      return 0;
    }
    ++cnt;
  }
}