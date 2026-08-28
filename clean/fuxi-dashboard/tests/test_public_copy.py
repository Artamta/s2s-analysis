"""Focused checks for public contact and release-facing terminology."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EMAIL = "raj.ayush@students.iiserpune.ac.in"
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_public_copy_contains_only_the_approved_contact_email() -> None:
    public_copy = "\n".join(
        read(path)
        for path in (
            "index.html",
            "src/main.ts",
            "src/pages/archive.ts",
            "src/pages/forecast.ts",
            "src/pages/global.ts",
            "src/pages/outlook.ts",
            "public/meeting.html",
        )
    )
    assert set(EMAIL_PATTERN.findall(public_copy)) == {PUBLIC_EMAIL}


def test_shared_footer_has_mail_and_institutional_links() -> None:
    shell = read("src/main.ts")
    assert f'href="mailto:{PUBLIC_EMAIL}"' in shell
    assert "https://scdlds.ashoka.edu.in/" in shell
    assert "Wednesday and Saturday" in shell
    assert "Limitations & contact" in shell


def test_dated_global_demo_copy_is_consistent() -> None:
    copy = "\n".join(
        read(path)
        for path in (
            "index.html",
            "README.md",
            "src/pages/global.ts",
        )
    )
    assert "global beta" not in copy.casefold()
    assert "dated global demo" in copy.casefold()
    assert f'href="mailto:{PUBLIC_EMAIL}"' in read("src/pages/global.ts")


def test_navigation_does_not_use_lab_branding() -> None:
    shell = read("src/main.ts")
    assert "S2S Lab" not in shell
    assert 'data-route="models"' not in shell


def test_primary_navigation_is_india_focused() -> None:
    shell = read("src/main.ts")
    assert 'data-route="india"' in shell
    assert 'data-route="archive"' in shell
    assert 'data-route="briefing"' in shell
    assert 'data-route="forecast"' not in shell
    assert 'data-route="outlook"' not in shell
    assert "?view=global" not in shell


def test_archive_exposes_an_exact_initialization_date_filter() -> None:
    archive = read("src/pages/archive.ts")
    assert 'id="archive-date-filter"' in archive
    assert '<option value="all">All dates</option>' in archive
    assert "issue.initialization === dateFilter.value" in archive
    assert "refreshDateOptions();" in archive


def test_member_count_is_kept_to_the_forecast_status_detail() -> None:
    forecast = read("src/pages/forecast.ts")
    archive = read("src/pages/archive.ts")
    briefing = read("src/pages/briefing.ts")
    assert forecast.count("${forecast.issue.members} members") == 1
    assert "member ensemble mean" not in forecast
    assert "Current 100-member proxy" not in forecast
    assert "${issue.members} members" not in archive
    assert "FEATURED_REPORT.ensemble" not in briefing


def test_featured_report_link_targets_a_valid_public_pdf() -> None:
    shell = read("src/main.ts")
    briefing = read("src/pages/briefing.ts")
    report_metadata = read("src/lib/reports.ts")
    report_path = "reports/s2s-summary-india-2026-08-19.pdf"
    cover_path = "reports/s2s-summary-india-2026-08-19-cover.png"
    report = ROOT / "public" / report_path
    cover = ROOT / "public" / cover_path
    assert f'href: "./{report_path}"' in report_metadata
    assert f'cover: "./{cover_path}"' in report_metadata
    assert 'href="${FEATURED_REPORT.href}"' in briefing
    assert 'src="${FEATURED_REPORT.href}"' in briefing
    assert 'type="application/pdf"' in briefing
    assert 'href="${FEATURED_REPORT.href}" download type="application/pdf"' in briefing
    assert 'aria-describedby="briefing-report-fallback"' in briefing
    assert "19 August 2026 (PDF, 2.1 MB)" in report_metadata
    assert report.read_bytes().startswith(b"%PDF-")
    assert cover.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_briefing_navigation_lists_the_scientific_team_by_surname() -> None:
    shell = read("src/main.ts")
    briefing = read("src/pages/briefing.ts")
    assert 'href="./#briefing" data-route="briefing"' in shell
    names_by_surname = (
        "Saptarishi Dhanuka",
        "Sandeep Juneja",
        "Parthasarathi Mukhopadhyay",
        "Ayush Raj",
        "Manmeet Singh",
    )
    for name in names_by_surname:
        assert name in briefing
    assert [briefing.index(name) for name in names_by_surname] == sorted(
        briefing.index(name) for name in names_by_surname
    )
    for scientific_role in (
        "Pre-doctoral Fellow · Forecast analysis",
        "Professor of Computer Science · Scientific computing",
        "Atmospheric scientist · Subseasonal forecasting",
        "BS–MS Student · Forecast analysis",
        "Project Lead · Subseasonal forecasting",
    ):
        assert scientific_role in briefing
    assert "publication workflow" not in briefing.casefold()
    assert "website maintenance" not in briefing.casefold()


def test_briefing_places_the_complete_report_before_the_team() -> None:
    briefing = read("src/pages/briefing.ts")
    summary_at = briefing.index('class="briefing-feature briefing-report-summary"')
    report_at = briefing.index('class="briefing-report"')
    team_at = briefing.index('class="briefing-team"')
    assert summary_at < report_at < team_at
    assert 'class="briefing-report__viewer"' in briefing
    assert 'title="Full ${FEATURED_REPORT.title} for ${FEATURED_REPORT.date}"' in briefing
    assert "Your browser cannot display embedded PDFs." in briefing
    assert "If the embedded report is unavailable," in briefing


def test_briefing_team_portraits_are_local_jpegs() -> None:
    portraits = (
        "ayush-raj.jpg",
        "sandeep-juneja.jpg",
        "parthasarathi-mukhopadhyay.jpg",
        "manmeet-singh.jpg",
        "saptarishi-dhanuka.jpg",
    )
    briefing = read("src/pages/briefing.ts")
    for filename in portraits:
        assert f'./team/{filename}' in briefing
        assert (ROOT / "public" / "team" / filename).read_bytes().startswith(b"\xff\xd8\xff")


def test_plan_is_explicitly_a_historical_record() -> None:
    plan = read("PLAN.md")
    assert plan.startswith("# Historical Design Record:")
    assert "It is not the current operations\n> plan or roadmap." in plan
    assert "Historical next milestones (superseded)" in plan
