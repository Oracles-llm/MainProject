# Desktop LLM Chat UI (ChatGPT-style)

A clean, modern AI chat interface — UI only with mock conversations. Light/dark theme support.

## Layout

**Left Sidebar (~260px, collapsible)**

- "New Chat" button at top (prominent, + icon)
- Scrollable list of previous chats below
- Each row: title + hover actions (rename / delete)
- Active chat highlighted
- Collapse toggle (collapses to narrow icon rail)

**Main Chat Area**

- Top bar: current chat title + theme toggle (sun/moon)
- Centered conversation column (readable max-width)
- User messages: right-aligned tinted bubble
- Assistant messages: left-aligned with avatar, subtle background
- Empty state: "How can I help you today?" + example prompt cards
- Bottom: rounded input with send button, attach icon, small "AI can make mistakes" disclaimer

## Interactions (mock, client-side state)

- New Chat → creates empty conversation, focuses input
- Send → appends user message + canned mock assistant reply with typing dots animation
- Rename chat → inline editable title
- Delete chat → with confirm dialog
- Theme toggle → switches light/dark
- Sidebar collapse → icon-only rail
- Mobile: sidebar becomes a drawer

## Visual Style

- Neutral grays, ChatGPT-like clean palette
- Inter font, generous spacing, subtle borders
- Smooth hover and transition animations

## Files

- `src/routes/index.tsx` — main chat page (replaces placeholder)
- `src/components/chat/Sidebar.tsx`
- `src/components/chat/ChatView.tsx`
- `src/components/chat/MessageBubble.tsx`
- `src/components/chat/EmptyState.tsx`
- `src/hooks/useChats.ts` — local chat/message state
- Seeded with ~5 sample conversations
