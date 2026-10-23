import { useEffect, useRef, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/core/react";
import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  ThreadPrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import { physicianNotice, shouldStartRun } from "../composer";
import { linkParts } from "../links";
import {
  parseInlines,
  parseMarkdown,
  splitAnswer,
  type Block,
  type Inline,
} from "../markdown";
import { formatMessageTime, rememberedMessageTime, rememberMessageTime } from "../messageTime";
import { historyTitle, useCompanionRuntime, writeThreadTitle } from "../runtime";
import { loadSession } from "../session";

type HomeProps = {
  onSignOut: () => void;
  username: string | null;
};

export function Home({ onSignOut, username }: HomeProps) {
  const runtime = useCompanionRuntime(onSignOut);
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="home">
        <AssistantSidebar onSignOut={onSignOut} username={username} />
        <section className="pane">
          <IntakeSeed onUnauthorized={onSignOut} />
          <HistoryTitle />
          <Thread />
          <p className="notice">{physicianNotice}</p>
        </section>
      </div>
    </AssistantRuntimeProvider>
  );
}

type MemoryItem = { id: string; text: string; field: string | null };

function authHeader(): { Authorization: string } | null {
  const session = loadSession();
  if (!session) return null;
  return { Authorization: `Bearer ${session.token}` };
}

function IntakeSeed({ onUnauthorized }: { onUnauthorized: () => void }) {
  const aui = useAui();
  const messageCount = useAuiState((state) => state.thread.messages.length);
  const remoteId = useAuiState((state) => state.threadListItem.remoteId);
  const seededFor = useRef<string | null>(null);
  useEffect(() => {
    if (messageCount > 0) return;
    const mark = remoteId ?? "new";
    if (seededFor.current === mark) return;
    const headers = authHeader();
    if (!headers) return;
    let cancelled = false;
    seededFor.current = mark;
    void (async () => {
      const response = await fetch("/api/intake", { headers });
      if (response.status === 401) {
        onUnauthorized();
        return;
      }
      if (!response.ok || cancelled) return;
      const body = (await response.json()) as { status?: string; next_field?: string | null };
      if (cancelled || body.status !== "pending" || body.next_field !== "age") return;
      aui.thread().append("INTAKE_START");
    })();
    return () => {
      cancelled = true;
    };
  }, [aui, messageCount, onUnauthorized, remoteId]);
  return null;
}

function HistoryTitle() {
  const aui = useAui();
  const remoteId = useAuiState((state) => state.threadListItem.remoteId);
  const signature = useAuiState((state) => state.thread.messages.map((message) => message.id).join("|"));
  const written = useRef("");
  useEffect(() => {
    if (!remoteId) return;
    const messages = aui.thread().getState().messages.map((message) => ({
      role: message.role,
      content: message.content,
    }));
    const title = historyTitle(messages);
    if (!title || title === written.current) return;
    written.current = title;
    void writeThreadTitle(remoteId, title);
  }, [aui, remoteId, signature]);
  return null;
}

function Memories({ onUnauthorized }: { onUnauthorized: () => void }) {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<MemoryItem[]>([]);

  useEffect(() => {
    if (!open) return;
    const headers = authHeader();
    if (!headers) return;
    let cancelled = false;
    void (async () => {
      const response = await fetch("/api/memories", { headers });
      if (response.status === 401) {
        onUnauthorized();
        return;
      }
      if (!response.ok || cancelled) return;
      const body = (await response.json()) as { items: MemoryItem[] };
      if (!cancelled) setItems(body.items);
    })();
    return () => {
      cancelled = true;
    };
  }, [open, onUnauthorized]);

  async function remove(id: string) {
    const headers = authHeader();
    if (!headers) return;
    const response = await fetch(`/api/memories/${encodeURIComponent(id)}`, {
      method: "DELETE",
      headers,
    });
    if (response.status === 401) {
      onUnauthorized();
      return;
    }
    if (response.ok) setItems((current) => current.filter((item) => item.id !== id));
  }

  return (
    <div className="memories">
      <button type="button" className="new-chat" onClick={() => setOpen((value) => !value)}>
        Memories
      </button>
      {open && items.length === 0 ? <p className="empty">No memories yet.</p> : null}
      {open && items.length > 0 ? (
        <ul className="memory-list">
          {items.map((item) => (
            <li key={item.id} className="memory-row">
              <span>{item.text}</span>
              <button type="button">Keep</button>
              <button type="button" onClick={() => void remove(item.id)}>
                Delete
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function AssistantSidebar({ onSignOut, username }: HomeProps) {
  const initial = username?.trim().slice(0, 1).toUpperCase();
  return (
    <aside className="sidebar">
      <p className="product">Clinical Companion</p>
      {username ? (
        <div className="who">
          <div className="avatar" aria-hidden="true">
            {initial}
          </div>
          <strong>{username}</strong>
        </div>
      ) : null}
      <Memories onUnauthorized={onSignOut} />
      <ThreadList onSignOut={onSignOut} />
    </aside>
  );
}

function ThreadList({ onSignOut }: { onSignOut: () => void }) {
  return (
    <ThreadListPrimitive.Root className="thread-list-root">
      <div className="thread-list">
        <ThreadListPrimitive.Items>
          {({ threadListItem }) =>
            threadListItem.title?.trim() ? (
              <ThreadListItemPrimitive.Root
                className="thread-item"
                data-thread={threadListItem.remoteId}
              >
                <ThreadListItemPrimitive.Trigger className="thread-trigger">
                  <ThreadListItemPrimitive.Title />
                </ThreadListItemPrimitive.Trigger>
              </ThreadListItemPrimitive.Root>
            ) : null
          }
        </ThreadListPrimitive.Items>
      </div>
      <div className="sidebar-actions">
        <ThreadListPrimitive.New className="new-chat">New chat</ThreadListPrimitive.New>
        <button type="button" className="sign-out" onClick={onSignOut}>
          Sign out
        </button>
      </div>
    </ThreadListPrimitive.Root>
  );
}

function LinkedText({ text }: { text: string }) {
  return (
    <>
      {linkParts(text).map((part, index) =>
        part.kind === "link" ? (
          <a key={index} href={part.url} target="_blank" rel="noopener noreferrer">
            {part.label}
          </a>
        ) : (
          <span key={index}>{part.text}</span>
        ),
      )}
    </>
  );
}

function InlineText({ inlines }: { inlines: Inline[] }) {
  return (
    <>
      {inlines.map((part, index) => {
        if (part.type === "link") {
          return (
            <a key={index} href={part.url} target="_blank" rel="noopener noreferrer">
              {part.label}
            </a>
          );
        }
        if (part.type === "bold") return <strong key={index}>{part.text}</strong>;
        if (part.type === "italic") return <em key={index}>{part.text}</em>;
        return <span key={index}>{part.text}</span>;
      })}
    </>
  );
}

function MarkdownBody({ blocks }: { blocks: Block[] }) {
  return (
    <div className="message-body">
      {blocks.map((block, index) => {
        if (block.type === "heading") {
          return (
            <p key={index} className={`message-heading message-heading-${block.level}`}>
              <InlineText inlines={block.inlines} />
            </p>
          );
        }
        if (block.type === "list") {
          return (
            <ul key={index}>
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}>
                  <InlineText inlines={item} />
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={index}>
            <InlineText inlines={block.inlines} />
          </p>
        );
      })}
    </div>
  );
}

function useVisitTime(id: string, createdAt: Date, live: boolean): Date | null {
  const [time, setTime] = useState<Date | null>(() => rememberedMessageTime(id));
  const millis = createdAt.getTime();
  useEffect(() => {
    const existing = rememberedMessageTime(id);
    if (existing) {
      setTime(existing);
      return;
    }
    if (!live) {
      setTime(null);
      return;
    }
    const stamp = new Date(millis);
    rememberMessageTime(id, stamp);
    setTime(stamp);
  }, [id, live, millis]);
  return time;
}

function MessageTime({ time }: { time: Date | null }) {
  if (!time) return null;
  return (
    <time className="message-time" dateTime={time.toISOString()}>
      {formatMessageTime(time)}
    </time>
  );
}

function messageText(content: readonly { type: string; text?: string }[]): string {
  return content.map((part) => (part.type === "text" ? part.text ?? "" : "")).join("");
}

function UserMessage() {
  const text = useAuiState((state) => messageText(state.message.content));
  const id = useAuiState((state) => state.message.id);
  const createdAt = useAuiState((state) => state.message.createdAt);
  const live = useAuiState((state) => isLiveMessage(state));
  const time = useVisitTime(id, createdAt, live);
  if (text.trim() === "INTAKE_START") return null;
  return (
    <MessagePrimitive.Root className="message message-user">
      <LinkedText text={text} />
      <MessageTime time={time} />
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  const text = useAuiState((state) => messageText(state.message.content));
  const status = useAuiState((state) => state.message.status?.type);
  const id = useAuiState((state) => state.message.id);
  const createdAt = useAuiState((state) => state.message.createdAt);
  const live = useAuiState((state) => isLiveMessage(state));
  const time = useVisitTime(id, createdAt, live);
  if (!text.trim()) {
    if (status !== "running") return null;
    return (
      <MessagePrimitive.Root className="message message-assistant message-pending">
        <span className="pending-dot" aria-hidden="true" />
        <span>Processing your request…</span>
        <MessageTime time={time} />
      </MessagePrimitive.Root>
    );
  }
  const answer = splitAnswer(text);
  return (
    <MessagePrimitive.Root className="message message-assistant">
      <MarkdownBody blocks={parseMarkdown(answer.body)} />
      {answer.sources.length > 0 ? (
        <div className="sources">
          <p className="sources-title">Sources</p>
          <ul>
            {answer.sources.map((source, index) => (
              <li key={index}>
                <InlineText inlines={parseInlines(source)} />
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <MessageTime time={time} />
    </MessagePrimitive.Root>
  );
}

function isLiveMessage(state: {
  message: { id: string; isLast: boolean; status?: { type: string } };
  thread: { isRunning: boolean; messages: readonly { id: string; role: string }[] };
}): boolean {
  if (state.message.status?.type === "running") return true;
  if (!state.thread.isRunning) return false;
  if (state.message.isLast) return true;
  for (let index = state.thread.messages.length - 1; index >= 0; index -= 1) {
    const message = state.thread.messages[index];
    if (message.role === "user") return message.id === state.message.id;
  }
  return false;
}

function Thread() {
  return (
    <ThreadPrimitive.Root className="thread">
      <ThreadPrimitive.Viewport className="viewport">
        <div className="transcript">
          <ThreadPrimitive.Empty>
            <p className="empty">Send a message to start.</p>
          </ThreadPrimitive.Empty>
          <ThreadPrimitive.Messages
            components={{
              UserMessage,
              AssistantMessage,
              SystemMessage: HiddenMessage,
            }}
          />
        </div>
        <ThreadPrimitive.ViewportFooter className="composer-footer">
          <Composer />
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
}

function HiddenMessage() {
  return null;
}

function Composer() {
  const text = useAuiState((state) => state.thread.composer.text);
  return (
    <ComposerPrimitive.Root className="composer">
      <ComposerPrimitive.Input placeholder="Message" rows={2} />
      {shouldStartRun(text) ? (
        <ComposerPrimitive.Send className="send">Send</ComposerPrimitive.Send>
      ) : (
        <button type="button" className="send" disabled>
          Send
        </button>
      )}
    </ComposerPrimitive.Root>
  );
}
