import java.util.List;
import java.util.Map;

@SuppressWarnings({"unused", "unchecked"})
public final class PlanningToolStubs {
    private PlanningToolStubs() {
        throw new AssertionError("Utility class");
    }

    /**
     * Convert file between formats (e.g., CSV to JSON, XML to YAML, etc.)
     *
     * Optional parameters: target_format.
     */
    public static Map<String, Object> convert_file_format(String input_path, String output_path, String target_format) {
        Object[] args = new Object[]{input_path, output_path, target_format};
        Object __result = PlanningToolRuntime.invoke("convert_file_format", args);
        return (Map<String, Object>) __result;
    }

    /**
     * Get detailed file information without reading content
     */
    public static Map<String, Object> get_file_info(String file_path) {
        Object[] args = new Object[]{file_path};
        Object __result = PlanningToolRuntime.invoke("get_file_info", args);
        return (Map<String, Object>) __result;
    }

    /**
     * Lists the contents of a specified directory (relative to project root).
     *
     * Optional parameters: directory_path.
     */
    public static Map<String, Object> list_directory(String directory_path) {
        Object[] args = new Object[]{directory_path};
        Object __result = PlanningToolRuntime.invoke("list_directory", args);
        return (Map<String, Object>) __result;
    }

    /**
     * Reads the content of a specified file.
     *
     * Optional parameters: num_lines.
     */
    public static Map<String, Object> read_file(String file_path, Integer num_lines) {
        Object[] args = new Object[]{file_path, num_lines};
        Object __result = PlanningToolRuntime.invoke("read_file", args);
        return (Map<String, Object>) __result;
    }

    /**
     * Read file content with advanced features like format detection, encoding detection, and
     * security validation.
     *
     * Optional parameters: encoding, format_hint.
     */
    public static Map<String, Object> read_file_advanced(String file_path, String encoding, String format_hint) {
        Object[] args = new Object[]{file_path, encoding, format_hint};
        Object __result = PlanningToolRuntime.invoke("read_file_advanced", args);
        return (Map<String, Object>) __result;
    }

    /**
     * Search for a text token within files under a repository-relative directory.
     *
     * Optional parameters: case_sensitive, max_results, include_hidden, allowed_extensions.
     */
    public static Map<String, Object> search_text_in_repository(String search_root, String query, Boolean case_sensitive, Integer max_results, Boolean include_hidden, List<String> allowed_extensions) {
        Object[] args = new Object[]{search_root, query, case_sensitive, max_results, include_hidden, allowed_extensions};
        Object __result = PlanningToolRuntime.invoke("search_text_in_repository", args);
        return (Map<String, Object>) __result;
    }

    /**
     * Writes content to a specified file (relative to project root).
     *
     * Optional parameters: mode.
     */
    public static Map<String, Object> write_file(String file_path, String content, String mode) {
        Object[] args = new Object[]{file_path, content, mode};
        Object __result = PlanningToolRuntime.invoke("write_file", args);
        return (Map<String, Object>) __result;
    }

    /**
     * Write content to file with advanced features like format detection, backup and atomic writes.
     *
     * Optional parameters: format_hint, encoding, backup.
     */
    public static Map<String, Object> write_file_advanced(String file_path, Object content, String format_hint, String encoding, Boolean backup) {
        Object[] args = new Object[]{file_path, content, format_hint, encoding, backup};
        Object __result = PlanningToolRuntime.invoke("write_file_advanced", args);
        return (Map<String, Object>) __result;
    }
}
