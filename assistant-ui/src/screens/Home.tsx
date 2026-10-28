import { useEffect, useId, useRef, useState, type ReactNode } from "react";
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
import {
  historyCreatedAt,
  historyTitle,
  useCompanionRuntime,
  whenThreadTitlesSettled,
} from "../runtime";
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
    try {
      void aui.threadListItem().rename(title);
    } catch {
      written.current = "";
    }
  }, [aui, remoteId, signature]);
  return null;
}

function Overlay({
  open,
  title,
  onClose,
  children,
}: {
  open: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open) {
      if (!dialog.open) dialog.showModal();
      return;
    }
    if (dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={dialogRef}
      className="overlay"
      aria-labelledby={titleId}
      onCancel={(event) => {
        event.preventDefault();
        onCloseRef.current();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) onCloseRef.current();
      }}
    >
      <div className="overlay-panel">
        <header className="overlay-header">
          <h2 id={titleId}>{title}</h2>
          <button type="button" className="overlay-close" onClick={() => onCloseRef.current()} aria-label="Close">
            Close
          </button>
        </header>
        <div className="overlay-body">{children}</div>
      </div>
    </dialog>
  );
}

function Memories({ onUnauthorized }: { onUnauthorized: () => void }) {
  const [open, setOpen] = useState(false);
  const [loaded, setLoaded] = useState(false);
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
      if (cancelled) return;
      if (!response.ok) {
        setLoaded(true);
        return;
      }
      const body = (await response.json()) as { items: MemoryItem[] };
      if (cancelled) return;
      setItems(body.items);
      setLoaded(true);
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
      <button type="button" className="sidebar-toggle" aria-haspopup="dialog" onClick={() => setOpen(true)}>
        <ButtonIcon name="memories" />
        My memories
      </button>
      <Overlay open={open} title="My memories" onClose={() => setOpen(false)}>
        {!loaded ? <p className="empty">Loading memories…</p> : null}
        {loaded && items.length === 0 ? <p className="empty">No memories yet.</p> : null}
        {loaded && items.length > 0 ? (
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
      </Overlay>
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
      <ThreadList onSignOut={onSignOut} />
    </aside>
  );
}

function ThreadList({ onSignOut }: { onSignOut: () => void }) {
  const [historyOpen, setHistoryOpen] = useState(false);
  const aui = useAui();
  const hasHistory = useAuiState((state) =>
    state.threads.threadItems.some((item) => Boolean(item.title?.trim())),
  );
  useEffect(() => {
    if (!historyOpen) return;
    let cancelled = false;
    void (async () => {
      await whenThreadTitlesSettled();
      if (!cancelled) await aui.threads().reload();
    })();
    return () => {
      cancelled = true;
    };
  }, [aui, historyOpen]);
  return (
    <ThreadListPrimitive.Root className="thread-list-root">
      <Memories onUnauthorized={onSignOut} />
      <button type="button" className="sidebar-toggle" aria-haspopup="dialog" onClick={() => setHistoryOpen(true)}>
        <ButtonIcon name="history" />
        History
      </button>
      <Overlay open={historyOpen} title="History" onClose={() => setHistoryOpen(false)}>
        {hasHistory ? null : <p className="empty">No chats yet.</p>}
        <div className="thread-list">
          <ThreadListPrimitive.Items>
            {({ threadListItem }) =>
              threadListItem.title?.trim() ? (
                <ThreadListItemPrimitive.Root
                  className="thread-item"
                  data-thread={threadListItem.remoteId}
                >
                  <ThreadListItemPrimitive.Trigger
                    className="thread-trigger"
                    onClick={() => setHistoryOpen(false)}
                  >
                    <ThreadListItemPrimitive.Title />
                    <ThreadCreated custom={threadListItem.custom} />
                  </ThreadListItemPrimitive.Trigger>
                </ThreadListItemPrimitive.Root>
              ) : null
            }
          </ThreadListPrimitive.Items>
        </div>
      </Overlay>
      <hr className="sidebar-separator" />
      <div className="sidebar-actions">
        <ThreadListPrimitive.New className="new-chat">
          <ButtonIcon name="new" />
          New chat
        </ThreadListPrimitive.New>
        <button type="button" className="sign-out" onClick={onSignOut}>
          <ButtonIcon name="sign-out" />
          Sign out
        </button>
      </div>
    </ThreadListPrimitive.Root>
  );
}

function ThreadCreated({ custom }: { custom?: Record<string, unknown> }) {
  const created = historyCreatedAt(custom);
  if (!created) return null;
  return (
    <time className="thread-created" dateTime={created.toISOString()}>
      {formatMessageTime(created)}
    </time>
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

function ButtonIcon({ name }: { name: "memories" | "history" | "new" | "send" | "sign-out" }) {
  return (
    <svg className="button-icon" viewBox="0 0 24 24" aria-hidden="true">
      {name === "memories" ? (
        <path d="M7 3h8l5 5v13H7zM15 3v5h5M9 13h6M9 17h6" />
      ) : null}
      {name === "history" ? <path d="M12 8v5l3 2M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18" /> : null}
      {name === "new" ? <path d="M12 5v14M5 12h14" /> : null}
      {name === "send" ? <path d="M4 12 20 4l-6 16-2.2-6.2L4 12z" /> : null}
      {name === "sign-out" ? <path d="M10 7V4h10v16H10v-3M4 12h11M12 9l3 3-3 3" /> : null}
    </svg>
  );
}

function Composer() {
  const text = useAuiState((state) => state.thread.composer.text);
  return (
    <ComposerPrimitive.Root className="composer">
      <ComposerPrimitive.Input placeholder="Message" rows={2} />
      {shouldStartRun(text) ? (
        <ComposerPrimitive.Send className="send">
          <ButtonIcon name="send" />
          Send
        </ComposerPrimitive.Send>
      ) : (
        <button type="button" className="send" disabled>
          <ButtonIcon name="send" />
          Send
        </button>
      )}
    </ComposerPrimitive.Root>
  );
}
