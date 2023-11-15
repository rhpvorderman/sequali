import array

from sequali import report_modules


def test_per_position_mean_quality_one_nuc():
    # https://github.com/rhpvorderman/sequali/issues/62
    # Would cause a divide by zero error
    module = report_modules.PerPositionMeanQualityAndSpread.from_phred_table_and_labels(
        array.array('Q', [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ['1']
    )
    assert module
