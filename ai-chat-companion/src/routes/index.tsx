import { createFileRoute } from "@tanstack/react-router";
import App from "@/App";

export const Route = createFileRoute("/")({
  component: App,
  head: () => ({
    meta: [
      { title: "Lumen — AI Chat Assistant" },
      {
        name: "description",
        content:
          "A clean, modern desktop chat interface for AI conversations with sidebar history.",
      },
    ],
  }),
});
