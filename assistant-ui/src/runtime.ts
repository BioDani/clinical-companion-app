import { useMemo } from "react";
import {
  useRemoteThreadListRuntime,
  type RemoteThreadListAdapter,
} from "@assistant-ui/react";
import {
  useLangGraphRuntime,
  type LangChainMessage,
} from "@assistant-ui/react-langgraph";
import { createAssistantStream } from "assistant-stream";
import { ASSISTANT_ID, createClient } from "./chatApi";
import { shouldStartRun } from "./composer";
import { loadSession } from "./session";

type ListedThread = {
  thread_id: string;
  metadata?: Record<string, unknown> | null;
};

function threadTitle(metadata: Record<string, unknown> | null | undefined): string | undefined {
  const title = metadata?.title;
  return typeof title === "string" && title.trim() ? title : undefined;
}

function messageText(message: LangChainMessage): string {
  const content = message.content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => (part.type === "text" || part.type === "text_delta" ? part.text : ""))
    .join("");
}

function lastHumanText(messages: readonly LangChainMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.type === "human") return messageText(message);
  }
  return "";
}

const EVIDENCE_PREFIX = "Use only the evidence below.";

const INTERNAL_LABELS = new Set([
  "GREETING",
  "IN_SCOPE",
  "OUT_OF_SCOPE",
  "SAFE",
  "UNSAFE",
]);

export function isVisibleTranscriptMessage(message: LangChainMessage): boolean {
  if (message.type !== "human" && message.type !== "ai") return false;
  const text = messageText(message).trim();
  if (INTERNAL_LABELS.has(text)) return false;
  if (text.startsWith(EVIDENCE_PREFIX)) return false;
  return true;
}

function visibleStreamData(data: unknown): unknown {
  if (!Array.isArray(data)) return data;
  return data.filter((message) => isVisibleTranscriptMessage(message as LangChainMessage));
}

function statusOf(error: unknown): number | undefined {
  if (typeof error === "object" && error !== null && "status" in error) {
    const status = (error as { status?: unknown }).status;
    if (typeof status === "number") return status;
  }
  return undefined;
}

function firstMessageTitle(
  messages: readonly { role: string; content: readonly { type: string; text?: string }[] }[],
): string {
  for (const message of messages) {
    if (message.role !== "user") continue;
    for (const part of message.content) {
      if (part.type === "text" && part.text?.trim()) return part.text.trim().slice(0, 80);
    }
  }
  return "";
}

/**
 * Stream one text run. The installed @assistant-ui/react-langgraph package
 * exports useLangGraphRuntime; the stream callback is this helper.
 */
export function createLangGraphStream(onUnauthorized: () => void) {
  return async function* stream(
    messages: LangChainMessage[],
    config: {
      abortSignal: AbortSignal;
      initialize: () => Promise<{ remoteId: string; externalId: string | undefined }>;
    },
  ) {
    if (!shouldStartRun(lastHumanText(messages))) return;
    const { externalId } = await config.initialize();
    if (!externalId) return;
    const session = loadSession();
    const client = createClient();
    try {
      const generator = client.runs.stream(externalId, ASSISTANT_ID, {
        input: { messages },
        streamMode: "messages",
        signal: config.abortSignal,
        metadata: {
          langfuse_session_id: externalId,
          langfuse_user_id: session?.userId,
          langfuse_tags: ["clinical-companion"],
        },
      });
      for await (const chunk of generator) {
        if (chunk.event === "messages/complete" || chunk.event === "messages/partial") {
          const data = visibleStreamData(chunk.data);
          if (Array.isArray(data) && data.length === 0) continue;
          yield { event: chunk.event, data };
          continue;
        }
        yield { event: chunk.event, data: chunk.data };
      }
      // Token streaming only carries what the model wrote. The agent adds the
      // "### Reference" / "### Sources" block after the model finishes, under
      // the same message id, so read the saved answer and replace it.
      if (!config.abortSignal.aborted) {
        const finished = await finalAnswer(client, externalId);
        if (finished.length > 0) yield { event: "messages/complete", data: finished };
      }
    } catch (error) {
      if (statusOf(error) === 401) onUnauthorized();
      throw error;
    }
  };
}

async function finalAnswer(
  client: ReturnType<typeof createClient>,
  threadId: string,
): Promise<LangChainMessage[]> {
  try {
    const state = await client.threads.getState<{ messages?: LangChainMessage[] }>(threadId);
    const messages = state.values.messages ?? [];
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message.type === "human") break;
      if (message.type === "ai" && isVisibleTranscriptMessage(message)) return [message];
    }
  } catch {
    // Keep the streamed answer if the saved state cannot be read.
  }
  return [];
}

function threadListAdapter(onUnauthorized: () => void): RemoteThreadListAdapter {
  return {
    async list() {
      try {
        const threads = (await createClient().threads.search({
          limit: 50,
          sortBy: "updated_at",
          sortOrder: "desc",
        })) as ListedThread[];
        return {
          threads: threads.flatMap((thread) => {
            const title = threadTitle(thread.metadata);
            if (!title) return [];
            return [
              {
                status: "regular" as const,
                remoteId: thread.thread_id,
                externalId: thread.thread_id,
                title,
              },
            ];
          }),
        };
      } catch (error) {
        if (statusOf(error) === 401) onUnauthorized();
        throw error;
      }
    },
    async initialize() {
      const thread = await createClient().threads.create({ graphId: ASSISTANT_ID });
      return { remoteId: thread.thread_id, externalId: thread.thread_id };
    },
    async fetch(threadId) {
      try {
        const thread = await createClient().threads.get(threadId);
        return {
          status: "regular" as const,
          remoteId: thread.thread_id,
          externalId: thread.thread_id,
          title: threadTitle(thread.metadata as Record<string, unknown> | undefined),
        };
      } catch (error) {
        if (statusOf(error) === 401) onUnauthorized();
        throw error;
      }
    },
    async rename() {
      return undefined;
    },
    async archive() {
      return undefined;
    },
    async unarchive() {
      return undefined;
    },
    async delete() {
      return undefined;
    },
    async generateTitle(remoteId, messages) {
      const title = firstMessageTitle(messages);
      if (title) {
        const client = createClient();
        const current = await client.threads.get(remoteId);
        await client.threads.update(remoteId, {
          metadata: { ...(current.metadata ?? {}), title },
        });
      }
      return createAssistantStream(() => undefined);
    },
  };
}

export function useCompanionRuntime(onUnauthorized: () => void) {
  const adapter = useMemo(() => threadListAdapter(onUnauthorized), [onUnauthorized]);
  const stream = useMemo(() => createLangGraphStream(onUnauthorized), [onUnauthorized]);
  return useRemoteThreadListRuntime({
    adapter,
    runtimeHook() {
      return useLangGraphRuntime({
        stream,
        load: async (threadId) => {
          try {
            const state = await createClient().threads.getState<{
              messages?: LangChainMessage[];
            }>(threadId);
            const messages = (state.values.messages ?? []).filter(isVisibleTranscriptMessage);
            return { messages };
          } catch (error) {
            if (statusOf(error) === 404) return { messages: [] };
            if (statusOf(error) === 401) onUnauthorized();
            throw error;
          }
        },
      });
    },
  });
}
