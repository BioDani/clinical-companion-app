import { AssistantRuntimeProvider } from "@assistant-ui/core/react";
import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  ThreadPrimitive,
  useAuiState,
} from "@assistant-ui/react";
import { physicianNotice, shouldStartRun } from "../composer";
import { linkParts } from "../links";
import { useCompanionRuntime } from "../runtime";

type HomeProps = {
  onSignOut: () => void;
};

export function Home({ onSignOut }: HomeProps) {
  const runtime = useCompanionRuntime(onSignOut);
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="home">
        <AssistantSidebar onSignOut={onSignOut} />
        <section className="pane">
          <Thread />
          <p className="notice">{physicianNotice}</p>
        </section>
      </div>
    </AssistantRuntimeProvider>
  );
}

function AssistantSidebar({ onSignOut }: HomeProps) {
  return (
    <aside className="sidebar">
      <p className="product">Clinical Companion</p>
      <ThreadList />
      <button type="button" className="sign-out" onClick={onSignOut}>
        Sign out
      </button>
    </aside>
  );
}

function ThreadList() {
  return (
    <ThreadListPrimitive.Root className="thread-list-root">
      <ThreadListPrimitive.New className="new-chat">New chat</ThreadListPrimitive.New>
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

function TextMessage({ variant }: { variant: "user" | "assistant" }) {
  const text = useAuiState((state) =>
    state.message.content
      .map((part) => (part.type === "text" ? part.text : ""))
      .join(""),
  );
  return (
    <MessagePrimitive.Root className={`message message-${variant}`}>
      <LinkedText text={text} />
    </MessagePrimitive.Root>
  );
}

function UserMessage() {
  return <TextMessage variant="user" />;
}

function AssistantMessage() {
  return <TextMessage variant="assistant" />;
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
