"use client";
import {
  PromptInput,
  PromptInputBody,
  PromptInputButton,
  PromptInputHeader,
  PromptInputTextarea,
  PromptInputProvider,
} from "../ai-elements/prompt-input";
import { Message } from "../ai-elements/message";
import { MessageResponse } from "../ai-elements/message";
import { SendIcon } from "lucide-react";
import { Conversation, ConversationContent } from "../ai-elements/conversation";
import { useState } from "react";

type Incident = {
  id: string;
  lat: number;
  lng: number;
  description: string;
  datetime: number;
  neighborhood: string;
};

const newId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random()}`;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export default function ChatBox({ incidents = [], summary = "" }: { incidents?: Incident[]; summary?: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: "welcome", role: "assistant", content: "Hello, how can I help you today?" },
  ]);
  const [pending, setPending] = useState(false);

  const handleFormSubmit = async (
    message: { text?: string; files?: unknown[] },
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();
    const userText = message.text?.trim();
    if (!userText) return;

    const compact = (incidents ?? [])
      .slice()
      .sort((a, b) => (b.datetime ?? 0) - (a.datetime ?? 0))
      .slice(0, 10)
      .map((i) => ({
        id: i.id,
        lat: i.lat,
        lng: i.lng,
        description: (i.description || "").slice(0, 120),
        datetime: i.datetime,
        neighborhood: i.neighborhood,
      }));

    let context = `Context for navigation and safety:\nSummary: ${summary || "No summary available"}\nIncidents (up to 10 most recent): ${JSON.stringify(
      compact
    )}\nInstructions: Use the provided data to answer questions about safety, recent incidents, and navigating the area. When giving directions, reference neighborhoods and relative positions (N/S/E/W) where helpful.`;
    const MAX_CONTEXT_CHARS = 4000;
    if (context.length > MAX_CONTEXT_CHARS) {
      context = context.slice(0, MAX_CONTEXT_CHARS) + "…";
    }

    setMessages((prev) => [...prev, { id: newId(), role: "user", content: userText }]);
    setPending(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: `${context}\n\nUser question: ${userText}`,
        }),
      });
      const data = await response.json();
      const reply = data.response || "I'm sorry, I don't have a response right now.";
      setMessages((prev) => [...prev, { id: newId(), role: "assistant", content: reply }]);
    } catch (error) {
      console.error("Error calling chat endpoint:", error);
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "assistant", content: "I encountered an error. Please try again." },
      ]);
    } finally {
      setPending(false);
    }
  };

  return (
    <PromptInputProvider initialInput="">
      <div className="flex h-[482px] w-full mx-auto max-w-2xl flex-col overflow-hidden rounded-2xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <Conversation className="flex-1 overflow-y-auto p-4">
          <ConversationContent>
            {messages.map((message) => {
              const isUser = message.role === "user";
              const bubbleColors = isUser
                ? "bg-blue-600 text-white border-blue-500/30"
                : "bg-zinc-100 text-zinc-900 border-zinc-300 dark:bg-zinc-800 dark:text-zinc-100 dark:border-zinc-700";
              return (
                <Message key={message.id} from={message.role} className="max-w-[85%]">
                  <div className={`w-fit rounded-2xl px-4 py-2 border shadow-sm ${bubbleColors}`}>
                    <MessageResponse>{message.content}</MessageResponse>
                  </div>
                </Message>
              );
            })}
          </ConversationContent>
        </Conversation>
        <div className="border-t border-zinc-200/70 bg-white p-3 dark:border-zinc-800/70 dark:bg-zinc-900">
          <PromptInput onSubmit={handleFormSubmit} className="w-full">
            <PromptInputHeader>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">Chat with Scrolink</p>
            </PromptInputHeader>
            <PromptInputBody>
              <div className="flex items-center justify-center gap-2 w-full">
                <PromptInputTextarea placeholder="Ask me about incidents in the area" className="w-full" disabled={pending} />
                <PromptInputButton
                  type="submit"
                  className="bg-blue-600 text-white hover:bg-blue-600/90"
                  disabled={pending}
                >
                  <SendIcon className="size-4" />
                </PromptInputButton>
              </div>
            </PromptInputBody>
          </PromptInput>
        </div>
      </div>
    </PromptInputProvider>
  );
}
