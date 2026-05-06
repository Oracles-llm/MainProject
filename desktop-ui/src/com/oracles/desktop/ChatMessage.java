package com.oracles.desktop;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

final class ChatMessage implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    enum Role {
        USER,
        ASSISTANT
    }

    private final String id;
    private final Role role;
    private final String content;

    ChatMessage(String id, Role role, String content) {
        this.id = Objects.requireNonNull(id);
        this.role = Objects.requireNonNull(role);
        this.content = Objects.requireNonNull(content);
    }

    String getId() {
        return id;
    }

    Role getRole() {
        return role;
    }

    String getContent() {
        return content;
    }
}
