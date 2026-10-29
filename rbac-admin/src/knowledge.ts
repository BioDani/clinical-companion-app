export type KnowledgeFile = {
  file_name: string;
  local_url: string;
  web_url: string;
  title: string;
  indexed: boolean;
};

export function localUrlFor(fileName: string): string {
  return `/knowledge/files/${fileName}`;
}

type KnowledgeRowInput = Omit<KnowledgeFile, "title" | "local_url" | "web_url"> & {
  local_url?: string;
  web_url?: string;
  title?: string;
};

export function catalogRows(
  payload: { files?: KnowledgeRowInput[] } | KnowledgeRowInput[] | null | undefined,
): KnowledgeFile[] {
  const files = Array.isArray(payload) ? payload : payload?.files ?? [];
  return files.map((file) => ({
    file_name: file.file_name,
    local_url: file.local_url || localUrlFor(file.file_name),
    web_url: file.web_url ?? "",
    title: file.title ?? "",
    indexed: Boolean(file.indexed),
  }));
}
