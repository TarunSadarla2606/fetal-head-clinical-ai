# Hadlock gestational-age estimation from head circumference

> Scope: the GA formula this system uses, its provenance, and how to read its
> uncertainty. The coefficients below are taken directly from
> `app/inference.py::hadlock_ga` and are accurate to this implementation.

## The Hadlock HC-to-GA formula as implemented here

> Source: `app/inference.py::hadlock_ga`, citing Hadlock FP et al.,
> AJR 1984;143:97–100.

Gestational age in weeks is computed from head circumference with the
polynomial

    GA = 8.96 + 0.540·x − 0.0040·x² + 0.000399·x³

where `x` is the head circumference **in centimetres** (the code divides the
millimetre HC by 10 before substituting). The result is clamped to the range
10–42 weeks and then formatted as whole weeks plus days.

## The formula is a population regression, not a per-fetus calculation

> Source: Hadlock FP et al., AJR 1984;143:97–100.

The polynomial describes the average relationship between head size and
gestational age across a reference population. It returns the gestational age
at which the observed HC is the population mean. It does not account for
constitutional variation in head size, ethnicity, fetal sex, or pathology, so
two fetuses of identical true gestational age but different head sizes will
receive different estimates.

## Uncertainty grows with gestational age

> Source: TODO(verbatim) — paste the specific prediction intervals from Hadlock
> 1984 or from the reference chart in local use. Do NOT state a numeric ± range
> here until sourced; the intervals differ by gestational window.

The scatter of true gestational age around the HC-predicted value widens as
pregnancy advances, because biological variation in fetal size accumulates. A
GA estimate derived from HC in the third trimester therefore carries a wider
prediction interval than the same calculation performed in the second
trimester. Any confidence interval quoted alongside a GA estimate must come
from the reference chart in use.

## Reference population and applicability

> Source: Hadlock FP et al., AJR 1984;143:97–100.
> TODO(verbatim): note any locally mandated alternative chart (e.g.
> INTERGROWTH-21st, WHO) if your institution does not use Hadlock.

The Hadlock 1984 equations were derived from a specific study population.
Applying them to a population with different growth characteristics introduces
systematic bias. Several later reference standards exist and institutions
differ in which they mandate. A GA estimate is only interpretable against the
chart that produced it, and charts should not be mixed within a report.

## Trimester boundaries used by this system

> Source: `app/inference.py::classify_trimester`, following ACOG / ISUOG / ACR
> conventions.

This system classifies gestational age as first trimester below 14 weeks 0
days, second trimester from 14 weeks 0 days through 27 weeks 6 days, and third
trimester from 28 weeks 0 days onward. The trimester shown with a result is
derived from the HC-estimated GA, so it inherits that estimate's uncertainty
and can differ from the trimester implied by established clinical dating.
