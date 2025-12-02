import java.util.List;
import java.util.Map;

@SuppressWarnings("unused")
public final class PlanningToolStubs {
    private PlanningToolStubs() {
        throw new AssertionError("Utility class");
    }

    /**
     * Apply a targetted text replacement inside an existing file.
     *
     * Optional parameters: occurrence, encoding.
     */
    public static Map<String, Object> apply_text_rewrite(String file_path, String original_snippet, String new_snippet, Integer occurrence, String encoding) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Convert file between formats (e.g., CSV to JSON, XML to YAML, etc.)
     *
     * Optional parameters: target_format.
     */
    public static Map<String, Object> convert_file_format(String input_path, String output_path, String target_format) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Get detailed file information without reading content
     */
    public static Map<String, Object> get_file_info(String file_path) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Clone a git repository, optionally targeting a branch and depth.
     *
     * Optional parameters: destination, branch, depth.
     */
    public static Map<String, Object> git_clone_repository(String repo_url, String destination, String branch, Integer depth) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Commit staged changes with the provided message.
     *
     * Optional parameters: author_name, author_email.
     */
    public static Map<String, Object> git_commit_changes(String repo_path, String message, String author_name, String author_email) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Create a new branch from the specified base.
     *
     * Optional parameters: base_branch.
     */
    public static Map<String, Object> git_create_branch(String repo_path, String branch_name, String base_branch) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Create a GitHub pull request.
     *
     * Optional parameters: base_branch, title, body, draft, token_env_var.
     */
    public static Map<String, Object> git_create_pull_request(String repository, String head_branch, String base_branch, String title, String body, Boolean draft, String token_env_var) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Return staged and unstaged diffs so an LLM can craft commit messages.
     */
    public static Map<String, Object> git_get_uncommitted_changes(String repo_path) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Push the specified branch to a remote.
     *
     * Optional parameters: remote_name, branch_name, set_upstream.
     */
    public static Map<String, Object> git_push_branch(String repo_path, String remote_name, String branch_name, Boolean set_upstream) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Stage files or directories (defaults to all changes).
     *
     * Optional parameters: paths.
     */
    public static Map<String, Object> git_stage_paths(String repo_path, Object paths) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Suggest a unique branch name based on the provided issue reference.
     *
     * Optional parameters: repo_path, issue_reference, prefix, max_suffix_attempts.
     */
    public static Map<String, Object> git_suggest_branch_name(String repo_path, String issue_reference, String prefix, Integer max_suffix_attempts) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Check out the given branch.
     *
     * Optional parameters: branch_name, branch.
     */
    public static Map<String, Object> git_switch_branch(String repo_path, String branch_name, String branch) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Lists the contents of a specified directory (relative to project root).
     *
     * Optional parameters: directory_path.
     */
    public static Map<String, Object> list_directory(String directory_path) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * List files below a root directory with optional glob filtering.
     *
     * Optional parameters: pattern, max_results, include_hidden, follow_symlinks.
     */
    public static Map<String, Object> list_files_in_tree(String root_path, String pattern, Integer max_results, Boolean include_hidden, Boolean follow_symlinks) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Replace or create a text file with the provided content.
     *
     * Optional parameters: encoding, create_directories, ensure_trailing_newline.
     */
    public static Map<String, Object> overwrite_text_file(String file_path, String content, String encoding, Boolean create_directories, Boolean ensure_trailing_newline) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Return only the first Qlty issue that matches the provided filters.
     *
     * Optional parameters: categories, statuses, levels, tools, base_url, token, token_env_var, timeout, max_retries, retry_delay.
     */
    public static Map<String, Object> qlty_get_first_issue(String owner_key_or_id, String project_key_or_id, Object categories, Object statuses, Object levels, Object tools, String base_url, String token, String token_env_var, Double timeout, Integer max_retries, Double retry_delay) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Reads the content of a specified file.
     *
     * Optional parameters: num_lines.
     */
    public static Map<String, Object> read_file(String file_path, Integer num_lines) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Read file content with advanced features like format detection, encoding detection, and
     * security validation.
     *
     * Optional parameters: encoding, format_hint.
     */
    public static Map<String, Object> read_file_advanced(String file_path, String encoding, String format_hint) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Read a UTF-8 (or user-provided encoding) text file for analysis.
     *
     * Optional parameters: encoding, max_characters.
     */
    public static Map<String, Object> read_text_file(String file_path, String encoding, Integer max_characters) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Search for a text token within files under a repository-relative directory.
     *
     * Optional parameters: case_sensitive, max_results, include_hidden, allowed_extensions.
     */
    public static Map<String, Object> search_text_in_repository(String search_root, String query, Boolean case_sensitive, Integer max_results, Boolean include_hidden, List<String> allowed_extensions) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Writes content to a specified file (relative to project root).
     *
     * Optional parameters: mode.
     */
    public static Map<String, Object> write_file(String file_path, String content, String mode) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }

    /**
     * Write content to file with advanced features like format detection, backup and atomic writes.
     *
     * Optional parameters: format_hint, encoding, backup.
     */
    public static Map<String, Object> write_file_advanced(String file_path, Object content, String format_hint, String encoding, Boolean backup) {
        throw new UnsupportedOperationException("Stub for planning only.");
    }
}
