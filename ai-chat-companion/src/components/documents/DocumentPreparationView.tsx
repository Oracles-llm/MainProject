import { useRef, useState } from "react";
import {
  CheckCircle2,
  FileText,
  Moon,
  Play,
  RefreshCcw,
  Settings2,
  Sun,
  Upload,
  X,
  AlertCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { useDoc } from "@/hooks/useDoc";

type Props = {
  theme: "light" | "dark";
  onToggleTheme: () => void;
};

const statusLabel = {
  queued: "Queued",
  chunking: "Chunking",
  rewriting: "Rewriting",
  saved: "Saved",
  failed: "Failed",
} as const;

function formatSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function DocumentPreparationView({ theme, onToggleTheme }: Props) {
  const {
    documents,
    chunkSize,
    chunkOverlap,
    parallelWorkers,
    jobId,
    isProcessing,
    error,
    preparedCount,
    pendingCount,
    totalChunks,
    averageProgress,
    addDocuments,
    removeDocument,
    resetQueue,
    startPreparation,
    setChunkSize,
    setChunkOverlap,
    setParallelWorkers,
  } = useDoc();

  const [rewriteEnabled, setRewriteEnabled] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <section className="flex h-full flex-1 flex-col bg-background">
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
        <div className="min-w-0">
          <h1 className="truncate text-sm font-semibold text-foreground">Document preparation</h1>
          <p className="truncate text-xs text-muted-foreground">
            Upload, chunk, rewrite, and stage documents for the local knowledge database.
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleTheme}
          aria-label="Toggle theme"
          className="h-8 w-8"
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto grid max-w-6xl gap-4 px-4 py-5 lg:grid-cols-[minmax(0,1fr)_340px]">
          <div className="space-y-4">
            {error && (
              <div className="flex gap-3 rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950">
                <AlertCircle className="h-5 w-5 shrink-0 text-red-600 dark:text-red-400" />
                <div>
                  <p className="text-sm font-medium text-red-800 dark:text-red-200">Error</p>
                  <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
                </div>
              </div>
            )}

            <div
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setIsDragging(false);
                addDocuments(event.dataTransfer.files);
              }}
              className={cn(
                "flex min-h-[220px] flex-col items-center justify-center rounded-lg border border-dashed border-border bg-card px-6 py-8 text-center transition-colors",
                isDragging && "border-ring bg-accent",
              )}
            >
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-md bg-secondary text-secondary-foreground">
                <Upload className="h-6 w-6" />
              </div>
              <h2 className="text-lg font-semibold text-card-foreground">Upload documents</h2>
              <p className="mt-2 max-w-xl text-sm text-muted-foreground">
                Drop PDFs, text files, Markdown, or notes here. Preparation will run offline.
              </p>
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                <Button 
                  onClick={() => fileInputRef.current?.click()} 
                  className="gap-2"
                  disabled={isProcessing}
                >
                  <Upload className="h-4 w-4" />
                  Choose files
                </Button>
                <Button
                  variant="outline"
                  className="gap-2"
                  disabled={documents.length === 0 || isProcessing}
                  onClick={startPreparation}
                >
                  <Play className="h-4 w-4" />
                  {isProcessing ? "Processing..." : "Start preparation"}
                </Button>
              </div>
              <Input
                ref={fileInputRef}
                type="file"
                multiple
                className="hidden"
                accept=".pdf,.txt,.md,.markdown,.docx"
                onChange={(event) => addDocuments(event.target.files)}
                disabled={isProcessing}
              />
            </div>

            <div className="rounded-lg border border-border bg-card">
              <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
                <div>
                  <h2 className="text-sm font-semibold text-card-foreground">Preparation queue</h2>
                  <p className="text-xs text-muted-foreground">
                    Files added here are ready for the processing pipeline.
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="gap-2"
                  disabled={documents.length === 0 || isProcessing}
                  onClick={resetQueue}
                >
                  <RefreshCcw className="h-4 w-4" />
                  Reset
                </Button>
              </div>

              {documents.length === 0 ? (
                <div className="flex min-h-[220px] items-center justify-center px-4 text-center text-sm text-muted-foreground">
                  No documents selected yet.
                </div>
              ) : (
                <div className="divide-y divide-border">
                  {documents.map((doc) => (
                    <div key={doc.id} className="grid gap-3 px-4 py-3 md:grid-cols-[1fr_120px_140px_32px] md:items-center">
                      <div className="flex min-w-0 items-center gap-3">
                        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-secondary text-secondary-foreground">
                          <FileText className="h-4 w-4" />
                        </div>
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium text-card-foreground">
                            {doc.name}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {formatSize(doc.size)} · {doc.chunks} chunks
                          </p>
                          {doc.error && (
                            <p className="text-xs text-red-600 dark:text-red-400">{doc.error}</p>
                          )}
                        </div>
                      </div>
                      <Badge
                        variant={doc.status === "saved" ? "default" : doc.status === "failed" ? "destructive" : "secondary"}
                        className="w-fit gap-1"
                      >
                        {doc.status === "saved" && <CheckCircle2 className="h-3 w-3" />}
                        {statusLabel[doc.status]}
                      </Badge>
                      <div className="min-w-0">
                        <Progress value={doc.progress} />
                        <p className="mt-1 text-xs text-muted-foreground">{doc.progress}%</p>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => removeDocument(doc.id)}
                        aria-label={`Remove ${doc.name}`}
                        disabled={isProcessing}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <aside className="space-y-4">
            <div className="rounded-lg border border-border bg-card p-4">
              <div className="mb-4 flex items-center gap-2">
                <Settings2 className="h-4 w-4 text-muted-foreground" />
                <h2 className="text-sm font-semibold text-card-foreground">Processing settings</h2>
              </div>
              <div className="space-y-5">
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-3">
                    <Label htmlFor="chunk-size">Chunk size</Label>
                    <span className="text-xs text-muted-foreground">{chunkSize} tokens</span>
                  </div>
                  <Slider
                    id="chunk-size"
                    min={300}
                    max={1600}
                    step={50}
                    value={[chunkSize]}
                    onValueChange={([value]) => setChunkSize(value)}
                    disabled={isProcessing}
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-3">
                    <Label htmlFor="chunk-overlap">Overlap</Label>
                    <span className="text-xs text-muted-foreground">{chunkOverlap} tokens</span>
                  </div>
                  <Slider
                    id="chunk-overlap"
                    min={0}
                    max={300}
                    step={20}
                    value={[chunkOverlap]}
                    onValueChange={([value]) => setChunkOverlap(value)}
                    disabled={isProcessing}
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-3">
                    <Label htmlFor="parallel-workers">Workers</Label>
                    <span className="text-xs text-muted-foreground">{parallelWorkers}</span>
                  </div>
                  <Slider
                    id="parallel-workers"
                    min={1}
                    max={8}
                    step={1}
                    value={[parallelWorkers]}
                    onValueChange={([value]) => setParallelWorkers(value)}
                    disabled={isProcessing}
                  />
                </div>

                <div className="flex items-center justify-between gap-4 rounded-md border border-border p-3">
                  <div>
                    <Label htmlFor="teacher-rewrite">Teacher rewrite</Label>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Improve chunks before saving them.
                    </p>
                  </div>
                  <Switch
                    id="teacher-rewrite"
                    checked={rewriteEnabled}
                    onCheckedChange={setRewriteEnabled}
                    disabled={isProcessing}
                  />
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-border bg-card p-4">
              <h2 className="text-sm font-semibold text-card-foreground">Run summary</h2>
              <div className="mt-4 grid grid-cols-2 gap-3">
                <SummaryStat label="Files" value={documents.length.toString()} />
                <SummaryStat label="Pending" value={pendingCount.toString()} />
                <SummaryStat label="Chunks" value={totalChunks.toString()} />
                <SummaryStat label="Saved" value={preparedCount.toString()} />
              </div>
              <div className="mt-4">
                <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
                  <span>Overall progress</span>
                  <span>{averageProgress}%</span>
                </div>
                <Progress value={averageProgress} />
              </div>
              {jobId && (
                <div className="mt-4 rounded-md bg-blue-50 p-2 dark:bg-blue-950">
                  <p className="text-xs text-blue-700 dark:text-blue-300">
                    Job ID: <span className="font-mono">{jobId}</span>
                  </p>
                </div>
              )}
            </div>
          </aside>
        </div>
      </div>
    </section>
  );
}

function SummaryStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 text-lg font-semibold text-foreground">{value}</p>
    </div>
  );
}
