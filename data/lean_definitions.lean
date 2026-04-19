-- Custom recursive definitions for Level 4+ theorems.
-- The training pipeline MUST load these into the Lean REPL
-- before checking any Level 4+ theorem.

def myDouble : Nat → Nat
  | 0 => 0
  | n + 1 => myDouble n + 2

def myPow2 : Nat → Nat
  | 0 => 1
  | n + 1 => 2 * myPow2 n

def myLen {α : Type _} : List α → Nat
  | [] => 0
  | _ :: xs => 1 + myLen xs

def myApp {α : Type _} : List α → List α → List α
  | [], ys => ys
  | x :: xs, ys => x :: myApp xs ys

def myRev {α : Type _} : List α → List α
  | [] => []
  | x :: xs => myApp (myRev xs) [x]

def myMap {α β : Type _} (f : α → β) : List α → List β
  | [] => []
  | x :: xs => f x :: myMap f xs

def myFilter {α : Type _} (p : α → Bool) : List α → List α
  | [] => []
  | x :: xs => if p x then x :: myFilter p xs else myFilter p xs

def myAll {α : Type _} (p : α → Bool) : List α → Bool
  | [] => true
  | x :: xs => p x && myAll p xs

def myIter {α : Type _} (f : α → α) : Nat → α → α
  | 0, x => x
  | n + 1, x => f (myIter f n x)
