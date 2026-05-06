import { Sparkles } from "lucide-react";

export function EmptyState(_: { onPick: (text: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6 py-12">
      <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
        <Sparkles className="h-7 w-7" />
      </div>
      <h2 className="text-2xl font-semibold tracking-tight text-foreground">
        How can I help you today?
      </h2>
      <p className="mt-2 text-sm text-muted-foreground">Ask anything to get started.</p>
    </div>
  );
}
