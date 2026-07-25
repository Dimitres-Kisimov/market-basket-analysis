from openpyxl import load_workbook

from basket.exports import write_deliverables


def test_deliverables_written_and_nonempty(tmp_path):
    sizes = write_deliverables(output_dir=str(tmp_path), n_baskets=1500, seed=42)
    paths = sorted(sizes)
    assert len(paths) == 2
    assert any(path.endswith(".pdf") for path in paths)
    assert any(path.endswith(".xlsx") for path in paths)
    for path, size in sizes.items():
        assert size > 10_000, f"{path} is only {size} bytes"

    excel_path = next(path for path in paths if path.endswith(".xlsx"))
    workbook = load_workbook(excel_path, read_only=True)
    assert workbook.sheetnames == ["Rules", "Itemsets", "Segments", "Recommendations"]
    workbook.close()
