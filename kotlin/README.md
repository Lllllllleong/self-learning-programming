# Kotlin LeetCode Solutions

This directory contains Kotlin solutions to LeetCode problems, organized by difficulty.

## 📁 Structure

```
.
├── easy/
│   └── Solutions.kt      # package easy
├── medium/
│   └── Solutions.kt      # package medium
├── hard/
│   └── Solutions.kt      # package hard
└── draft.kt              # Working file (gitignored, in root)
```

## ▶️ Running

Write a solution plus a `main()` in `draft.kt` (repo root), then:

```powershell
kotlinc draft.kt -include-runtime -d out.jar; java -jar out.jar
```

Move the finished solution (without `main()`) into the matching difficulty file.

> Note: package names can't start with `kotlin` (reserved), so each file uses just `easy` / `medium` / `hard`.
