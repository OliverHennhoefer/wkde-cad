# Interpreting \(-\log_{10}(p)\), \(\log_{10}(m/\alpha)\), and BH/FDR Detectability Boundaries

## 1. Meaning of \(\log_{10}(1/\text{minimal p-value})\)

The expression is likely:

\[
\log_{10}\left(\frac{1}{p_{\min}}\right)
\]

This is the same as:

\[
-\log_{10}(p_{\min})
\]

Because p-values are between 0 and 1, taking \(1/p\) makes small p-values large, and then \(\log_{10}\) makes the scale easier to read.

### Meaning of values

| \(-\log_{10}(p_{\min})\) | Minimal p-value |
|---:|---:|
| 1 | 0.1 |
| 2 | 0.01 |
| 3 | 0.001 |
| 4 | 0.0001 |
| 5 | 0.00001 |
| 6 | 0.000001 |

So a value of \(3\) means:

\[
p_{\min} = 10^{-3} = 0.001
\]

A value of \(5\) means:

\[
p_{\min} = 10^{-5} = 0.00001
\]

### General rule

If the plotted value is \(y\), then:

\[
p_{\min} = 10^{-y}
\]

Examples:

| \(-\log_{10}(p)\) | p-value |
|---:|---:|
| 1.3 | about 0.05 |
| 2 | 0.01 |
| 3 | 0.001 |
| 10 | \(10^{-10}\) |

So higher values mean smaller p-values, i.e. stronger statistical evidence.

---

## 2. Meaning of \(\log_{10}(m/\alpha)\)

If the quantity is:

\[
\log_{10}\left(\frac{m}{\alpha}\right)
\]

where:

- \(m\) is the batch size, number of tests, or number of comparisons
- \(\alpha\) is the significance level, often 0.05

then the value tells you the base-10 order of magnitude of \(m/\alpha\).

General rule:

\[
x = \log_{10}\left(\frac{m}{\alpha}\right)
\]

means:

\[
\frac{m}{\alpha} = 10^x
\]

Equivalently:

\[
\alpha = \frac{m}{10^x}
\]

or:

\[
m = \alpha \cdot 10^x
\]

### Values 1, 2, 3, 4, 5

| \(\log_{10}(m/\alpha)\) | \(m/\alpha\) |
|---:|---:|
| 1 | 10 |
| 2 | 100 |
| 3 | 1,000 |
| 4 | 10,000 |
| 5 | 100,000 |
| 6 | 1,000,000 |

Each increase by 1 means \(m/\alpha\) is 10 times larger.

### Example with \(\alpha = 0.05\)

If \(\alpha = 0.05\), then dividing by \(\alpha\) is the same as multiplying by 20.

| \(m\) | \(m/\alpha\) | \(\log_{10}(m/\alpha)\) |
|---:|---:|---:|
| 1 | 20 | 1.30 |
| 10 | 200 | 2.30 |
| 100 | 2,000 | 3.30 |
| 1,000 | 20,000 | 4.30 |
| 10,000 | 200,000 | 5.30 |

---

## 3. Significance threshold in relation to batch size and alpha

For a Bonferroni-style correction, the p-value threshold is:

\[
p_{\text{threshold}} = \frac{\alpha}{m}
\]

This controls the batch-level error rate at \(\alpha\) across \(m\) tests.

On the \(-\log_{10}(p)\) scale, this threshold becomes:

\[
-\log_{10}\left(\frac{\alpha}{m}\right)
=
\log_{10}\left(\frac{m}{\alpha}\right)
\]

So the decision rule is:

\[
-\log_{10}(p_{\min}) \geq \log_{10}\left(\frac{m}{\alpha}\right)
\]

Equivalently:

\[
p_{\min} \leq \frac{\alpha}{m}
\]

### Example with \(\alpha = 0.05\)

| Batch size \(m\) | p-value threshold \(\alpha/m\) | \(-\log_{10}\) threshold |
|---:|---:|---:|
| 1 | 0.05 | 1.30 |
| 10 | 0.005 | 2.30 |
| 100 | 0.0005 | 3.30 |
| 1,000 | 0.00005 | 4.30 |
| 10,000 | 0.000005 | 5.30 |

Example:

If \(m = 100\) and \(\alpha = 0.05\), then:

\[
\log_{10}(m/\alpha)
=
\log_{10}(100/0.05)
=
\log_{10}(2000)
\approx 3.30
\]

So the minimal p-value is significant only if:

\[
-\log_{10}(p_{\min}) \geq 3.30
\]

which is equivalent to:

\[
p_{\min} \leq 0.0005
\]

As batch size increases, the required p-value becomes smaller, and the required \(-\log_{10}(p)\) value becomes larger.

---

## 4. Heatmap interpretation: feasibility or detectability regions

Suppose your heatmap has:

\[
y = -\log_{10}(p_{\min})
\]

and:

\[
x = \log_{10}(m/\alpha)
\]

For a Bonferroni-style threshold, the decision rule is:

\[
-\log_{10}(p_{\min}) \geq \log_{10}(m/\alpha)
\]

So on the heatmap:

\[
y \geq x
\]

means the batch contains a statistically significant result after correction.

### Interpretation

| Region | Meaning |
|---|---|
| \(y > x\) | detectable / significant |
| \(y = x\) | exactly at threshold |
| \(y < x\) | not significant after correction |

The diagonal line:

\[
y = x
\]

is the significance boundary.

Above the diagonal: feasible / detectable.

Below the diagonal: not detectable under that \(m,\alpha\) threshold.

### Useful derived quantity

The vertical distance from the diagonal is:

\[
y - x
\]

That equals:

\[
-\log_{10}(p_{\min})
-
\log_{10}(m/\alpha)
\]

which can be rewritten as:

\[
-\log_{10}\left(\frac{p_{\min}}{\alpha/m}\right)
\]

So:

| \(y-x\) | Meaning |
|---:|---|
| positive | p-value is below threshold |
| zero | exactly at threshold |
| negative | p-value is above threshold |

Example:

If \(y = 5\) and \(x = 3\), then:

\[
y - x = 2
\]

That means the observed minimal p-value is \(10^2 = 100\) times smaller than the required threshold.

If \(y = 2\) and \(x = 4\), then:

\[
y - x = -2
\]

The p-value is \(10^2 = 100\) times too large to pass the threshold.

So the heatmap depicts feasibility / detectability regions, with the diagonal \(y=x\) as the critical boundary under a Bonferroni-style rule.

---

## 5. Is the decision line strict?

For the Bonferroni-style decision rule, yes: the line is a strict mathematical boundary.

The rule is:

\[
-\log_{10}(p_{\min}) \geq \log_{10}\left(\frac{m}{\alpha}\right)
\]

If your axes are:

\[
y = -\log_{10}(p_{\min})
\]

and:

\[
x = \log_{10}(m/\alpha)
\]

then the boundary is exactly:

\[
y = x
\]

### Decision regions

| Region | Decision |
|---|---|
| \(y > x\) | significant after correction |
| \(y = x\) | exactly at threshold |
| \(y < x\) | not significant after correction |

Using the usual convention:

\[
p_{\min} \leq \frac{\alpha}{m}
\]

the rule is:

\[
y \geq x
\]

So the line itself counts as significant if the test uses “less than or equal to.” In practice, equality is rare for continuous p-values, but it can occur with discrete or rounded p-values.

### Is it valid for all x/y combinations?

Algebraically, yes: for any valid \(x\) and \(y\), the line \(y=x\) separates passing from failing.

Statistically, it is valid only if:

1. \(m\) is the number of tests in the batch.
2. \(\alpha\) is the desired batch-level significance level.
3. \(p_{\min}\) is the smallest p-value from that same batch.
4. You are using a Bonferroni-style correction:

\[
p_{\text{threshold}} = \frac{\alpha}{m}
\]

Under those conditions, the boundary is valid across the whole plot.

### Important nuance

The line is not a universal detectability boundary in all statistical senses. It is specifically the Bonferroni significance boundary.

If you use another correction method, the line changes.

| Method | Boundary |
|---|---|
| Bonferroni | \(p_{\min} \leq \alpha/m\) |
| Šidák, independent tests | \(p_{\min} \leq 1-(1-\alpha)^{1/m}\) |
| FDR / Benjamini-Hochberg | threshold depends on rank, not just \(p_{\min}\) |
| Permutation / empirical correction | threshold estimated from null distribution |

So the diagonal \(y=x\) is the correct decision line for Bonferroni-corrected minimal p-values. It is valid over the full plotted domain as long as the x- and y-values are defined from the same \(m\), \(\alpha\), and batch of p-values.

---

## 6. In the FDR / Benjamini-Hochberg context

In the FDR / Benjamini-Hochberg context, the simple line:

\[
y = x
\]

is not the full decision boundary if:

\[
y = -\log_{10}(p_{\min})
\]

and:

\[
x = \log_{10}(m/\alpha)
\]

because BH does not only compare the minimal p-value to \(\alpha/m\).

BH first sorts the p-values:

\[
p_{(1)} \leq p_{(2)} \leq \dots \leq p_{(m)}
\]

Then it finds the largest rank \(k\) such that:

\[
p_{(k)} \leq \frac{k\alpha}{m}
\]

Then it rejects all hypotheses with ranks:

\[
1,\dots,k
\]

On the \(-\log_{10}\) scale, the BH threshold at rank \(k\) is:

\[
-\log_{10}(p_{(k)})
\geq
\log_{10}\left(\frac{m}{k\alpha}\right)
\]

### Relation to the diagonal line

For \(k=1\), BH uses:

\[
p_{(1)} \leq \frac{\alpha}{m}
\]

which is the same as Bonferroni.

So the line:

\[
-\log_{10}(p_{\min})
=
\log_{10}(m/\alpha)
\]

is the rank-1 BH boundary.

But BH can still reject even if the smallest p-value does not pass that line.

### Example

Let:

\[
m = 100
\]

and:

\[
\alpha = 0.05
\]

The Bonferroni / rank-1 threshold is:

\[
\frac{\alpha}{m} = 0.0005
\]

Suppose:

\[
p_{(1)} = 0.0008
\]

and:

\[
p_{(2)} = 0.0009
\]

The minimum p-value does not pass the rank-1 threshold:

\[
0.0008 > 0.0005
\]

But the rank-2 BH threshold is:

\[
\frac{2\alpha}{m}
=
\frac{2 \cdot 0.05}{100}
=
0.001
\]

and:

\[
p_{(2)} = 0.0009 \leq 0.001
\]

So BH rejects both hypotheses.

In a 2D plot using only \(p_{\min}\), this point would fall below the Bonferroni diagonal, but BH would still call discoveries.

### Correct interpretation

A heatmap with:

\[
y = -\log_{10}(p_{\min})
\]

and:

\[
x = \log_{10}(m/\alpha)
\]

shows a conservative detectability region for BH.

| Region | Meaning under BH |
|---|---|
| \(y \geq x\) | definitely at least one BH discovery |
| \(y < x\) | not enough information; BH may still reject if multiple low p-values exist |

So for BH/FDR, the line \(y=x\) is sufficient but not necessary.

### Better BH-specific quantity

For BH, the relevant comparison is rank-dependent:

\[
-\log_{10}(p_{(k)})
\quad
\text{vs.}
\quad
\log_{10}\left(\frac{m}{k\alpha}\right)
\]

Equivalently, define a BH margin:

\[
D_k =
-\log_{10}(p_{(k)})
-
\log_{10}\left(\frac{m}{k\alpha}\right)
\]

Then:

| \(D_k\) | Meaning |
|---:|---|
| \(D_k > 0\) | rank \(k\) passes BH |
| \(D_k = 0\) | exactly at BH threshold |
| \(D_k < 0\) | rank \(k\) does not pass |

BH chooses the largest \(k\) with:

\[
D_k \geq 0
\]

So the detectability boundary is not one fixed diagonal in the \((p_{\min}, m/\alpha)\) plane. It is a family of rank-dependent boundaries, shifted by \(\log_{10}(k)\).

---

## 7. Why the observed BH/FDR boundary can level off

In a BH/FDR heatmap, it is expected that the actual boundary may level off while the Bonferroni diagonal keeps rising.

The diagonal keeps rising because it corresponds to the Bonferroni / rank-1 threshold:

\[
p_{(1)} \leq \frac{\alpha}{m}
\]

or equivalently:

\[
-\log_{10}(p_{\min})
\geq
\log_{10}(m/\alpha)
\]

That threshold becomes stricter as \(m\) increases.

But BH uses:

\[
p_{(k)} \leq \frac{k\alpha}{m}
\]

The important factor is \(k\), the number of discoveries.

If many hypotheses have small p-values, then \(k\) can grow with \(m\).

Suppose:

\[
k \approx r m
\]

where \(r\) is the discovered fraction.

Then the BH threshold becomes:

\[
\frac{k\alpha}{m}
\approx
\frac{rm\alpha}{m}
=
r\alpha
\]

The \(m\) cancels.

So on the \(-\log_{10}\) scale, the effective threshold becomes approximately:

\[
-\log_{10}(r\alpha)
\]

which is roughly constant in \(m\).

That explains the leveling-off pattern.

### Intuition

Bonferroni asks:

> Is the single smallest p-value tiny enough to survive \(m\) tests?

BH asks:

> Is there a sufficiently large group of small p-values to justify discoveries while controlling FDR?

So if the signal is not just one isolated tiny p-value, but a cluster or fraction of low p-values, BH becomes much less stringent than Bonferroni.

The boundary can flatten because evidence accumulates across ranks.

### Why alpha and m still matter

The plateau depends on:

- \(\alpha\)
- the fraction of low p-values
- the shape of the p-value distribution
- the number of true or near-true signals

For example, if the effective discovery fraction is:

\[
r = 0.1
\]

and:

\[
\alpha = 0.05
\]

then the rough BH threshold is:

\[
r\alpha = 0.005
\]

so:

\[
-\log_{10}(0.005) \approx 2.30
\]

That gives a horizontal-ish boundary around:

\[
y \approx 2.30
\]

even if the Bonferroni diagonal continues rising.

### When the boundary would not level off

The leveling is expected when the number of real or near-real signals grows with batch size.

If instead you increase \(m\) mostly by adding null tests, while the number of true signals stays fixed, then \(k\) does not grow much with \(m\).

In that case:

\[
\frac{k\alpha}{m}
\]

still shrinks roughly like \(1/m\), so the boundary will rise more like the diagonal, though shifted downward by:

\[
\log_{10}(k)
\]

### Bottom line

For BH/FDR, a leveling boundary is normal.

It means the effective BH cutoff is being driven by rank mass / signal density, not only by the total number of tests.

The diagonal \(y=x\) is the conservative rank-1 boundary.

The actual BH frontier can flatten when many p-values jointly support rejection.