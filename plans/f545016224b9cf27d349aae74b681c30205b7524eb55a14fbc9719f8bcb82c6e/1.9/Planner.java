public class Planner {
    import java.util.List;
import java.util.Map;public class Planner {
    public static void main(String[] args) {
        if (hasOpenIssues()) {
            // Handle the issue
        } else {
            System.out.println("No open issues to address.");
        }
    }
        List<String> extensions = List.of(".json");
    }
    private static boolean hasOpenIssues() {
        List<String> extensions = List.of(".json");
        Map<String, Object> result = PlanningToolStubs.search_text_in_repository("/qlty/issues", "open", true, null, false, extensions);
        return result.get("found") != null;
    }
}
        return PlanningToolStubs.search_text_in_repository("/qlty/issues", "open", true, null, false, extensions).get("found") != null;
    }
}
