"""Tests for interpreting Java plans into actionable sub-goals."""
from __future__ import annotations

from textwrap import dedent

from llmflow.planning.java_plan_executor import JavaPlanNavigator


_SAMPLE_PLAN = dedent(
    """
    public class Planner {
        public static void main(String[] args) {
            repairLintIssue();
        }

        public static void repairLintIssue() {
            // Step 1: Prepare the workspace and create a feature branch
            PlanningToolStubs.createBranch();

            // Step 2: Clone or checkout the repository
            PlanningToolStubs.cloneRepo();

            // Step 3: Retrieve the first open lint issue from Qlty tools
            PlanningToolStubs.applyTextRewrite();
            PlanningToolStubs.qltyGetFirstIssue();
            String lintIssueId = PlanningToolStubs.readTextFile("queries/qlty_llvm-lint_results.md");

            // Step 4: Understand the issue details for the parse error on line 7
            PlanningToolStubs.readTextFile("diagnostics/qlty_issue_summary.md");

            // Step 5: Apply changes to resolve the parse error
            PlanningToolStubs.applyTextRewrite();

            // Step 6: Verify the fix with tests and commit evidence
            PlanningToolStubs.getUncommittedChanges();
            PlanningToolStubs.commitChanges();
            PlanningToolStubs.pushBranch();

            // Step 7: Finalize by opening a pull request summarizing the delivery
            PlanningToolStubs.createPullRequest();
        }
    }
    """
)


def test_navigator_extracts_first_subgoal_comment() -> None:
    navigator = JavaPlanNavigator.from_source(
        _SAMPLE_PLAN,
        tool_stub_class_name="PlanningToolStubs",
    )
    intent = navigator.next_subgoal()

    assert intent is not None
    assert intent.action_kind == "tool"
    assert intent.action_name == "createBranch"
    assert intent.parent_function == "repairLintIssue"
    assert intent.goal.lower().startswith("step 1")
    assert "create a feature branch" in intent.goal.lower()