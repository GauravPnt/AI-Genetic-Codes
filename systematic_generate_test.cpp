#include <bits/stdc++.h>
using namespace std;

int n;
enum { E, Q };

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

bool check(int x, int y) { return x < n && y < n && x >= 0 && y >= 0; }

bool test(vector<vector<int>> &board, int row, int col) {
  for (int j = 0; j < n; ++j) {
    if (board[j][col] == Q)
      return false;
    if (check(row - col + j, j) && board[row - col + j][j] == Q)
      return false;
    if (check(row + col - j, j) && board[row + col - j][j] == Q)
      return false;
  }
  return true;
}

int ctr = 0;

bool solve(vector<vector<int>> board, int row) {
  if (row == n) {
    cout << ctr << '\n';
    print(board);
    return true;
  }
  for (int col = 0; col < n; ++col) {
    if (!test(board, row, col)) {
      ++ctr;
      continue;
    }
    board[row][col] = Q;
    if (solve(board, row + 1))
      return true;
    board[row][col] = E;
  }
  return false;
}

int main() {
  cin >> n;
  vector<vector<int>> board(n, vector<int>(n));
  solve(board, 0);
}