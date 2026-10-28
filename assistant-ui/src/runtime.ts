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
  created_at?: string;
  metadata?: Record<string, unknown> | null;
};

/** Creation time stored on a history row. Null when the thread has no stamp. */
export function historyCreatedAt(custom: Record<string, unknown> | undefined): Date | null {
  const value = custom?.createdAt;
  if (typeof value !== "string") return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function threadCustom(createdAt: string | undefined): { createdAt: string } | undefined {
  const date = historyCreatedAt(createdAt ? { createdAt } : undefined);
  return date ? { createdAt: date.toISOString() } : undefined;
}

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

export const HEALTH_INTRODUCTION = "Health introduction";

const INTAKE_CLOSE = "Thank you. I will use what you chose to share on later habit questions.";

const INTAKE_HINTS = [
  "I’m wellnest, your wellness companion",
  "What is your age?",
  "What is your weight?",
  "What is your height?",
  "How many hours do you sleep at night?",
  "How many meals do you eat per day?",
  "What kind of diet do you follow?",
  "Is there a condition that should change suggestions",
  INTAKE_CLOSE,
];

type TitleMessage = {
  role: string;
  content: string | readonly { type: string; text?: string }[];
};

function titleText(message: TitleMessage): string {
  const content = message.content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((part) => (part.type === "text" ? part.text ?? "" : "")).join("");
}

function intakeAssistant(text: string): boolean {
  return INTAKE_HINTS.some((hint) => text.includes(hint));
}

/** Short history name from the visible conversation. Empty when the chat has no subject yet. */
export function historyTitle(messages: readonly TitleMessage[]): string {
  let expectingAnswer = false;
  let closed = false;
  let habit = "";
  for (const message of messages) {
    const text = titleText(message).trim();
    if (!text || text === "INTAKE_START") continue;
    const role = message.role;
    if (role === "assistant" || role === "ai") {
      if (!intakeAssistant(text)) {
        expectingAnswer = false;
        continue;
      }
      closed = text.includes(INTAKE_CLOSE);
      expectingAnswer = !closed;
      continue;
    }
    if (role !== "user" && role !== "human") continue;
    if (expectingAnswer) {
      expectingAnswer = false;
      continue;
    }
    if (!habit) habit = text.slice(0, 80);
  }
  if (habit) return habit;
  if (closed) return HEALTH_INTRODUCTION;
  return "";
}

type HistoryRow = {
  status: "regular";
  remoteId: string;
  externalId: string;
  title: string;
  custom?: { createdAt: string };
};

type RememberedThread = {
  title: string;
  custom?: { createdAt: string };
};

const rememberedThreads = new Map<string, RememberedThread>();
const pendingTitleWrites = new Set<Promise<void>>();

/** Threads titled in this session, kept when a history reload has not caught up. */
export function withRememberedThreads(
  listed: readonly HistoryRow[],
  remembered: ReadonlyMap<string, RememberedThread> = rememberedThreads,
): HistoryRow[] {
  const seen = new Set(listed.map((thread) => thread.remoteId));
  const missing: HistoryRow[] = [];
  for (const [remoteId, saved] of remembered) {
    if (seen.has(remoteId) || !saved.title.trim()) continue;
    missing.push({
      status: "regular",
      remoteId,
      externalId: remoteId,
      title: saved.title,
      custom: saved.custom,
    });
  }
  return [...missing, ...listed];
}

/** Resolves when every in-flight title save has finished. */
export function whenThreadTitlesSettled(): Promise<void> {
  return Promise.all([...pendingTitleWrites]).then(() => undefined);
}

export async function writeThreadTitle(remoteId: string, title: string): Promise<void> {
  const task = persistThreadTitle(remoteId, title);
  const settled = task.then(
    () => undefined,
    () => undefined,
  );
  pendingTitleWrites.add(settled);
  void settled.finally(() => pendingTitleWrites.delete(settled));
  return task;
}

async function persistThreadTitle(remoteId: string, title: string): Promise<void> {
  const client = createClient();
  const current = await client.threads.get(remoteId);
  if (current.metadata?.title !== title) {
    await client.threads.update(remoteId, {
      metadata: { ...(current.metadata ?? {}), title },
    });
  }
  rememberedThreads.set(remoteId, {
    title,
    custom: threadCustom(current.created_at),
  });
}

const INTERNAL_LABELS = new Set([
  "GREETING",
  "IN_SCOPE",
  "OUT_OF_SCOPE",
  "SAFE",
  "UNSAFE",
  "NONE",
  "INTAKE_START",
  "GROUNDED",
  "UNGROUNDED",
]);

const RELEVANCE_LABEL = /^RELEVANT(?:\s+\d+)*$/;

export function isVisibleTranscriptMessage(message: LangChainMessage): boolean {
  if (message.type !== "human" && message.type !== "ai") return false;
  const text = messageText(message).trim();
  if (INTERNAL_LABELS.has(text) || RELEVANCE_LABEL.test(text)) return false;
  if (text.startsWith(EVIDENCE_PREFIX)) return false;
  return true;
}

function visibleStreamData(data: unknown): unknown {
  if (!Array.isArray(data)) return data;
  return data.filter((message) => isVisibleTranscriptMessage(message as LangChainMessage));
}

/** Messages a node returned after the model draft, such as the citation footer. */
export function messagesFromUpdates(data: unknown): LangChainMessage[] {
  if (!data || typeof data !== "object") return [];
  const found: LangChainMessage[] = [];
  for (const update of Object.values(data as Record<string, unknown>)) {
    if (!update || typeof update !== "object") continue;
    const messages = (update as { messages?: unknown }).messages;
    if (!Array.isArray(messages)) continue;
    for (const message of messages) {
      if (isVisibleTranscriptMessage(message as LangChainMessage)) {
        found.push(message as LangChainMessage);
      }
    }
  }
  return found;
}

function statusOf(error: unknown): number | undefined {
  if (typeof error === "object" && error !== null && "status" in error) {
    const status = (error as { status?: unknown }).status;
    if (typeof status === "number") return status;
  }
  return undefined;
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
        streamMode: ["messages", "updates"],
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
        if (chunk.event === "updates") {
          const cited = messagesFromUpdates(chunk.data);
          if (cited.length > 0) {
            yield { event: "messages/complete", data: cited };
          }
          const data = chunk.data as { __interrupt__?: unknown } | null;
          if (data && typeof data === "object" && "__interrupt__" in data) {
            yield { event: chunk.event, data: chunk.data };
          }
          continue;
        }
        yield { event: chunk.event, data: chunk.data };
      }
    } catch (error) {
      if (statusOf(error) === 401) onUnauthorized();
      throw error;
    }
  };
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
        const listed = threads.flatMap((thread) => {
          const title = threadTitle(thread.metadata);
          if (!title) return [];
          return [
            {
              status: "regular" as const,
              remoteId: thread.thread_id,
              externalId: thread.thread_id,
              title,
              custom: threadCustom(thread.created_at),
            },
          ];
        });
        return { threads: withRememberedThreads(listed) };
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
          custom: threadCustom(thread.created_at),
        };
      } catch (error) {
        if (statusOf(error) === 401) onUnauthorized();
        throw error;
      }
    },
    async rename(remoteId, newTitle) {
      await writeThreadTitle(remoteId, newTitle);
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
      const title = historyTitle(messages);
      if (!title) return createAssistantStream(() => undefined);
      await writeThreadTitle(remoteId, title);
      return createAssistantStream((controller) => {
        controller.appendText(title);
      });
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
