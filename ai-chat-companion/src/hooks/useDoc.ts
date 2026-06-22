import { useCallback, useEffect, useState, useRef } from "react";

export type PrepStatus = "queued" | "chunking" | "rewriting" | "saved" | "failed";

export type PrepDocument = {
  id: string;
  name: string;
  size: number;
  status: PrepStatus;
  progress: number;
  chunks: number;
  file?: File;
  error?: string;
};

export type DocumentPrepJob = {
  jobId: string | null;
  isProcessing: boolean;
  error: string | null;
};

function getApiBaseUrl() {
  if (typeof window !== "undefined") {
    const apiBaseUrl = new URLSearchParams(window.location.search).get("apiBaseUrl");
    if (apiBaseUrl) return apiBaseUrl.replace(/\/$/, "");
  }

  return import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";
}

const API_BASE_URL = getApiBaseUrl();

async function uploadDocuments(
  files: File[],
  chunkSize: number,
  chunkOverlap: number,
  parallelWorkers: number,
  chunkingMethod: string,
): Promise<string> {
  const formData = new FormData();

  files.forEach((file) => {
    formData.append("files", file);
  });

  formData.append("chunk_size", chunkSize.toString());
  formData.append("chunk_overlap", chunkOverlap.toString());
  formData.append("parallel_workers", parallelWorkers.toString());
  formData.append("chunking_method", chunkingMethod);

  const response = await fetch(`${API_BASE_URL}/api/v1/documents/prepare`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
  }

  const data = await response.json();
  return data.job_id;
}

async function getJobStatus(jobId: string): Promise<{
  status: string;
  documents: Record<
    string,
    {
      status: string;
      chunks: number;
      error?: string;
    }
  >;
}> {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents/status/${jobId}`);

  if (!response.ok) {
    throw new Error(`Failed to get job status: ${response.statusText}`);
  }

  return await response.json();
}

async function cleanupJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents/cleanup/${jobId}`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Failed to cleanup job: ${response.statusText}`);
  }
}

export function useDoc() {
  const [documents, setDocuments] = useState<PrepDocument[]>([]);
  const [chunkSize, setChunkSize] = useState(900);
  const [chunkOverlap, setChunkOverlap] = useState(120);
  const [parallelWorkers, setParallelWorkers] = useState(4);
  const [chunkingMethod, setChunkingMethod] = useState("recursive");
  const [availableChunkingMethods, setAvailableChunkingMethods] = useState<string[]>(["recursive"]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const preparedCount = documents.filter((doc) => doc.status === "saved").length;
  const pendingCount = documents.length - preparedCount;
  const totalChunks = documents.reduce((sum, doc) => sum + doc.chunks, 0);
  const averageProgress =
    documents.length === 0 ? 0 : Math.round(documents.reduce((sum, doc) => sum + doc.progress, 0) / documents.length);

  const addDocuments = useCallback((fileList: FileList | null) => {
    if (!fileList?.length) return;

    const newDocs = Array.from(fileList).map((file) => ({
      id: Math.random().toString(36).slice(2, 10),
      name: file.name,
      size: file.size,
      status: "queued" as PrepStatus,
      progress: 0,
      chunks: Math.max(1, Math.ceil(file.size / 12000)),
      file,
    }));

    setDocuments((current) => [...newDocs, ...current]);
    setError(null);
  }, []);

  const removeDocument = useCallback((id: string) => {
    setDocuments((current) => current.filter((doc) => doc.id !== id));
  }, []);

  const resetQueue = useCallback(() => {
    setDocuments((current) => current.map((doc) => ({ ...doc, status: "queued" as PrepStatus, progress: 0, error: undefined })));
    setJobId(null);
    setError(null);
  }, []);

  const pollJobStatus = useCallback(async (currentJobId: string) => {
    try {
      const status = await getJobStatus(currentJobId);

      setDocuments((current) =>
        current.map((doc) => {
          const docStatus = status.documents[doc.name];
          if (!docStatus) return doc;

          let newStatus: PrepStatus = "queued";
          if (docStatus.status === "completed") {
            newStatus = "saved";
          } else if (docStatus.status === "failed") {
            newStatus = "failed";
          } else {
            newStatus = "chunking";
          }

          return {
            ...doc,
            status: newStatus,
            progress: newStatus === "saved" ? 100 : newStatus === "failed" ? 0 : 50,
            chunks: docStatus.chunks || doc.chunks,
            error: docStatus.error,
          };
        }),
      );

      // Stop polling if job is complete
      if (status.status === "completed" || status.status === "failed") {
        setIsProcessing(false);
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      }
    } catch (err) {
      console.error("Error polling job status:", err);
      setError(`Error checking progress: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  }, []);

  const startPreparation = useCallback(async () => {
    if (documents.length === 0) {
      setError("No documents selected");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const filesToUpload = documents.filter((doc) => doc.file).map((doc) => doc.file!);

      const uploadedJobId = await uploadDocuments(filesToUpload, chunkSize, chunkOverlap, parallelWorkers, chunkingMethod);
      setJobId(uploadedJobId);

      // Update documents to show they're processing
      setDocuments((current) =>
        current.map((doc) => ({
          ...doc,
          status: "chunking" as PrepStatus,
          progress: 10,
        })),
      );

      // Start polling for status
      const interval = setInterval(() => {
        pollJobStatus(uploadedJobId);
      }, 2000);

      pollingIntervalRef.current = interval;
    } catch (err) {
      setIsProcessing(false);
      const errorMessage = err instanceof Error ? err.message : "Unknown error occurred";
      setError(errorMessage);
      console.error("Error starting preparation:", err);
    }
  }, [documents, chunkSize, chunkOverlap, parallelWorkers, chunkingMethod, pollJobStatus]);

  const cleanup = useCallback(async () => {
    if (!jobId) return;

    try {
      await cleanupJob(jobId);
      setJobId(null);
    } catch (err) {
      console.error("Error cleaning up job:", err);
    }
  }, [jobId]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Ingested documents (already in Qdrant)
  // ---------------------------------------------------------------------------

  type IngestedDocument = {
    source: string;
    chunks: number;
  };

  const [ingestedDocuments, setIngestedDocuments] = useState<IngestedDocument[]>([]);
  const [ingestedLoading, setIngestedLoading] = useState(false);

  const fetchIngestedDocuments = useCallback(async () => {
    setIngestedLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/documents/list`);
      if (!response.ok) throw new Error(`Failed to fetch: ${response.statusText}`);
      const data = await response.json();
      setIngestedDocuments(data.documents ?? []);
    } catch (err) {
      console.error("Error fetching ingested documents:", err);
    } finally {
      setIngestedLoading(false);
    }
  }, []);

  const removeIngestedDocument = useCallback(async (sourceName: string) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/documents/remove/${encodeURIComponent(sourceName)}`,
        { method: "DELETE" },
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Delete failed: ${response.statusText}`);
      }
      // Remove from local state immediately
      setIngestedDocuments((current) => current.filter((d) => d.source !== sourceName));
    } catch (err) {
      console.error("Error removing ingested document:", err);
      setError(`Failed to remove document: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  }, []);

  // Fetch ingested documents and chunking methods on mount
  useEffect(() => {
    fetchIngestedDocuments();

    // Fetch available chunking methods
    fetch(`${API_BASE_URL}/api/v1/documents/chunking-methods`)
      .then((res) => res.json())
      .then((data) => {
        if (data.methods) setAvailableChunkingMethods(data.methods);
        if (data.default) setChunkingMethod(data.default);
      })
      .catch((err) => console.error("Error fetching chunking methods:", err));
  }, [fetchIngestedDocuments]);

  // Re-fetch after processing finishes
  const prevIsProcessingRef = useRef(isProcessing);
  useEffect(() => {
    if (prevIsProcessingRef.current && !isProcessing) {
      // Processing just finished — refresh the list
      fetchIngestedDocuments();
    }
    prevIsProcessingRef.current = isProcessing;
  }, [isProcessing, fetchIngestedDocuments]);

  return {
    // State
    documents,
    chunkSize,
    chunkOverlap,
    parallelWorkers,
    jobId,
    isProcessing,
    error,

    // Computed values
    preparedCount,
    pendingCount,
    totalChunks,
    averageProgress,

    // Actions
    addDocuments,
    removeDocument,
    resetQueue,
    startPreparation,
    cleanup,
    setChunkSize,
    setChunkOverlap,
    setParallelWorkers,
    chunkingMethod,
    setChunkingMethod,
    availableChunkingMethods,

    // Ingested documents
    ingestedDocuments,
    ingestedLoading,
    fetchIngestedDocuments,
    removeIngestedDocument,
  };
}
