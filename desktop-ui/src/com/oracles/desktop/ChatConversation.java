package com.oracles.desktop;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

final class ChatConversation implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private final String id;
    private String title;
    private final List<ChatMessage> messages;

    ChatConversation(String id, String title, List<ChatMessage> messages) {
        this.id = Objects.requireNonNull(id);
        this.title = Objects.requireNonNull(title);
        this.messages = new ArrayList<>(Objects.requireNonNull(messages));
    }

    String getId() {
        return id;
    }

    String getTitle() {
        return title;
    }

    void setTitle(String title) {
        this.title = Objects.requireNonNull(title);
    }

    List<ChatMessage> getMessages() {
        return messages;
    }
}
