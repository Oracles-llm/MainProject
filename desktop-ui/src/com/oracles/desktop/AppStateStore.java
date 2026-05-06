package com.oracles.desktop;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

final class AppStateStore {
    private final Path stateFile;

    AppStateStore() {
        this.stateFile = Path.of(
            System.getProperty("user.home"),
            ".oracles-desktop",
            "desktop-state.bin"
        );
    }

    PersistedState load() {
        if (!Files.isRegularFile(stateFile)) {
            return null;
        }

        try (ObjectInputStream input = new ObjectInputStream(Files.newInputStream(stateFile))) {
            Object value = input.readObject();
            if (value instanceof PersistedState state) {
                return state;
            }
        } catch (Exception ignored) {
            return null;
        }

        return null;
    }

    void save(List<ChatConversation> chats, String activeChatId, boolean darkTheme, boolean collapsed) {
        try {
            Files.createDirectories(stateFile.getParent());
            try (ObjectOutputStream output = new ObjectOutputStream(Files.newOutputStream(stateFile))) {
                output.writeObject(new PersistedState(new ArrayList<>(chats), activeChatId, darkTheme, collapsed));
            }
        } catch (IOException ignored) {
            // Local state persistence is optional.
        }
    }

    static final class PersistedState implements Serializable {
        @Serial
        private static final long serialVersionUID = 1L;

        private final List<ChatConversation> chats;
        private final String activeChatId;
        private final boolean darkTheme;
        private final boolean collapsed;

        PersistedState(List<ChatConversation> chats, String activeChatId, boolean darkTheme, boolean collapsed) {
            this.chats = chats;
            this.activeChatId = activeChatId;
            this.darkTheme = darkTheme;
            this.collapsed = collapsed;
        }

        List<ChatConversation> getChats() {
            return chats;
        }

        String getActiveChatId() {
            return activeChatId;
        }

        boolean isDarkTheme() {
            return darkTheme;
        }

        boolean isCollapsed() {
            return collapsed;
        }
    }
}
