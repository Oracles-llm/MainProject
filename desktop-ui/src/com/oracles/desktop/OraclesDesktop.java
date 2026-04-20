package com.oracles.desktop;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.RenderingHints;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Objects;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import javax.swing.UIManager;
import javax.swing.border.EmptyBorder;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

public final class OraclesDesktop {
    private static final Color TEXT_PRIMARY = new Color(245, 240, 230);
    private static final Color TEXT_MUTED = new Color(190, 186, 178);
    private static final Color PANEL_DARK = new Color(18, 18, 22, 230);
    private static final Color ASSISTANT_PANEL = new Color(18, 22, 24, 220);
    private static final Color USER_PANEL = new Color(64, 51, 31, 235);
    private static final Color ACCENT_AQUA = new Color(104, 224, 207);
    private static final Color ACCENT_GOLD = new Color(249, 212, 143);

    private final HttpClient httpClient;
    private final String apiBaseUrl;
    private final JPanel messagesPanel;
    private final JScrollPane scrollPane;
    private final JTextArea inputArea;
    private final JButton sendButton;
    private final JLabel statusLabel;
    private final JLabel heroTitle;
    private boolean loading;

    private OraclesDesktop(String apiBaseUrl) {
        this.apiBaseUrl = Objects.requireNonNull(apiBaseUrl);
        this.httpClient = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();

        this.messagesPanel = new JPanel();
        this.messagesPanel.setOpaque(false);
        this.messagesPanel.setLayout(new BoxLayout(this.messagesPanel, BoxLayout.Y_AXIS));
        this.messagesPanel.setBorder(new EmptyBorder(0, 0, 8, 0));

        this.scrollPane = new JScrollPane(messagesPanel);
        this.scrollPane.setBorder(null);
        this.scrollPane.getViewport().setOpaque(false);
        this.scrollPane.setOpaque(false);
        this.scrollPane.getVerticalScrollBar().setUnitIncrement(16);

        this.inputArea = new JTextArea(3, 40);
        this.inputArea.setLineWrap(true);
        this.inputArea.setWrapStyleWord(true);
        this.inputArea.setOpaque(false);
        this.inputArea.setForeground(TEXT_PRIMARY);
        this.inputArea.setCaretColor(TEXT_PRIMARY);
        this.inputArea.setFont(new Font("Dialog", Font.PLAIN, 16));
        this.inputArea.setBorder(BorderFactory.createEmptyBorder());
        this.inputArea.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent event) {
                refreshComposerState();
            }

            @Override
            public void removeUpdate(DocumentEvent event) {
                refreshComposerState();
            }

            @Override
            public void changedUpdate(DocumentEvent event) {
                refreshComposerState();
            }
        });
        this.inputArea.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent event) {
                if (event.getKeyCode() == KeyEvent.VK_ENTER && !event.isShiftDown()) {
                    event.consume();
                    sendMessage();
                }
            }
        });

        this.sendButton = createSendButton();
        this.statusLabel = new JLabel("Connected to " + apiBaseUrl);
        this.statusLabel.setForeground(TEXT_MUTED);
        this.statusLabel.setFont(new Font("Dialog", Font.PLAIN, 13));

        this.heroTitle = new JLabel("What are you working on?");
        this.heroTitle.setForeground(TEXT_PRIMARY);
        this.heroTitle.setFont(new Font("Dialog", Font.BOLD, 34));

        refreshComposerState();
    }

    public static void main(String[] args) {
        String apiBaseUrl = args.length > 0 ? args[0] : "http://localhost:8000";
        SwingUtilities.invokeLater(() -> {
            configureLookAndFeel();
            new OraclesDesktop(apiBaseUrl).show();
        });
    }

    private static void configureLookAndFeel() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {
            // Falling back to the default look and feel is acceptable here.
        }
    }

    private void show() {
        JFrame frame = new JFrame("Oracles LLM");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setMinimumSize(new Dimension(900, 700));
        frame.setContentPane(buildContent());
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private JComponent buildContent() {
        GradientPanel root = new GradientPanel();
        root.setLayout(new BorderLayout(0, 22));
        root.setBorder(new EmptyBorder(24, 28, 24, 28));

        root.add(buildHeader(), BorderLayout.NORTH);
        root.add(buildCenter(), BorderLayout.CENTER);
        root.add(buildComposer(), BorderLayout.SOUTH);
        return root;
    }

    private JComponent buildHeader() {
        JPanel header = new JPanel(new BorderLayout());
        header.setOpaque(false);

        JPanel brandRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 0));
        brandRow.setOpaque(false);

        JLabel brand = new JLabel("Oracles LLM");
        brand.setForeground(TEXT_PRIMARY);
        brand.setFont(new Font("Dialog", Font.BOLD, 22));

        JLabel pill = new JLabel("THINKING", SwingConstants.CENTER);
        pill.setOpaque(true);
        pill.setBackground(new Color(66, 84, 79, 190));
        pill.setForeground(TEXT_PRIMARY);
        pill.setFont(new Font("Dialog", Font.BOLD, 11));
        pill.setBorder(new EmptyBorder(4, 10, 4, 10));

        brandRow.add(brand);
        brandRow.add(pill);
        header.add(brandRow, BorderLayout.WEST);
        return header;
    }

    private JComponent buildCenter() {
        JPanel center = new JPanel();
        center.setOpaque(false);
        center.setLayout(new BorderLayout(0, 20));

        center.add(buildHeroPanel(), BorderLayout.NORTH);
        center.add(buildChatPanel(), BorderLayout.CENTER);
        return center;
    }

    private JComponent buildHeroPanel() {
        JPanel hero = new JPanel();
        hero.setOpaque(false);
        hero.setLayout(new BoxLayout(hero, BoxLayout.Y_AXIS));
        hero.setBorder(new EmptyBorder(8, 0, 0, 0));

        JLabel kicker = new JLabel("STUDIO MODE");
        kicker.setAlignmentX(Component.CENTER_ALIGNMENT);
        kicker.setForeground(TEXT_MUTED);
        kicker.setFont(new Font("Dialog", Font.BOLD, 12));

        heroTitle.setAlignmentX(Component.CENTER_ALIGNMENT);

        JLabel subtitle = new JLabel(
            "<html><div style='text-align:center;'>Ask your local assistant anything. "
                + "Responses appear below as soon as the model finishes.</div></html>"
        );
        subtitle.setAlignmentX(Component.CENTER_ALIGNMENT);
        subtitle.setForeground(TEXT_MUTED);
        subtitle.setFont(new Font("Dialog", Font.PLAIN, 15));

        hero.add(kicker);
        hero.add(Box.createVerticalStrut(10));
        hero.add(heroTitle);
        hero.add(Box.createVerticalStrut(10));
        hero.add(subtitle);
        return hero;
    }

    private JComponent buildChatPanel() {
        RoundedPanel shell = new RoundedPanel(new BorderLayout(), PANEL_DARK, new Color(255, 255, 255, 22));
        shell.setBorder(new EmptyBorder(18, 18, 18, 18));
        shell.add(scrollPane, BorderLayout.CENTER);
        return shell;
    }

    private JComponent buildComposer() {
        RoundedPanel composer = new RoundedPanel(new BorderLayout(0, 14), PANEL_DARK, new Color(255, 255, 255, 30));
        composer.setBorder(new EmptyBorder(16, 18, 16, 18));

        composer.add(inputArea, BorderLayout.CENTER);

        JPanel bottomRow = new JPanel(new BorderLayout());
        bottomRow.setOpaque(false);
        bottomRow.add(statusLabel, BorderLayout.WEST);
        bottomRow.add(sendButton, BorderLayout.EAST);

        composer.add(bottomRow, BorderLayout.SOUTH);
        return composer;
    }

    private JButton createSendButton() {
        JButton button = new JButton("Send");
        button.setFocusPainted(false);
        button.setBorderPainted(false);
        button.setOpaque(false);
        button.setForeground(new Color(28, 27, 25));
        button.setBackground(ACCENT_AQUA);
        button.setFont(new Font("Dialog", Font.BOLD, 15));
        button.setBorder(new EmptyBorder(10, 22, 10, 22));
        button.setContentAreaFilled(false);
        button.addActionListener(event -> sendMessage());
        return button;
    }

    private void sendMessage() {
        String content = inputArea.getText().trim();
        if (content.isEmpty() || loading) {
            return;
        }

        inputArea.setText("");
        addMessage("You", content, USER_PANEL);
        setLoading(true, null);

        HttpRequest request = HttpRequest.newBuilder(URI.create(apiBaseUrl + "/api/v1/chat"))
            .timeout(Duration.ofMinutes(5))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString("{\"query\":\"" + escapeJson(content) + "\"}"))
            .build();

        new SwingWorker<String, Void>() {
            @Override
            protected String doInBackground() throws Exception {
                HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
                if (response.statusCode() < 200 || response.statusCode() >= 300) {
                    throw new IllegalStateException(extractErrorMessage(response.body(), response.statusCode()));
                }
                return extractResponse(response.body());
            }

            @Override
            protected void done() {
                try {
                    String responseText = get();
                    addMessage("Oracles", responseText, ASSISTANT_PANEL);
                    setLoading(false, "Connected to " + apiBaseUrl);
                } catch (Exception error) {
                    setLoading(false, "Request failed");
                    JOptionPane.showMessageDialog(
                        null,
                        error.getCause() != null ? error.getCause().getMessage() : error.getMessage(),
                        "Request Error",
                        JOptionPane.ERROR_MESSAGE
                    );
                }
            }
        }.execute();
    }

    private void setLoading(boolean value, String statusText) {
        loading = value;
        sendButton.setText(value ? "Thinking..." : "Send");
        statusLabel.setText(statusText != null ? statusText : "Waiting for Oracles to respond...");
        heroTitle.setText(messagesPanel.getComponentCount() > 0 ? "Continue the conversation" : "What are you working on?");
        refreshComposerState();
    }

    private void addMessage(String role, String text, Color bubbleColor) {
        if (messagesPanel.getComponentCount() > 0) {
            messagesPanel.add(Box.createVerticalStrut(14));
        }
        messagesPanel.add(new MessageBubble(role, text, bubbleColor));
        messagesPanel.revalidate();
        scrollToBottom();
        inputArea.requestFocusInWindow();
        refreshComposerState();
    }

    private void refreshComposerState() {
        sendButton.setEnabled(!loading && !inputArea.getText().trim().isEmpty());
    }

    private void scrollToBottom() {
        SwingUtilities.invokeLater(() -> {
            var scrollBar = scrollPane.getVerticalScrollBar();
            scrollBar.setValue(scrollBar.getMaximum());
        });
    }

    private static String extractResponse(String body) {
        String marker = "\"response\":";
        int markerIndex = body.indexOf(marker);
        if (markerIndex < 0) {
            return "No response returned.";
        }
        int startQuote = body.indexOf('"', markerIndex + marker.length());
        if (startQuote < 0) {
            return "No response returned.";
        }
        int endQuote = findStringEnd(body, startQuote + 1);
        return unescapeJson(body.substring(startQuote + 1, endQuote));
    }

    private static String extractErrorMessage(String body, int statusCode) {
        if (body == null || body.isBlank()) {
            return "Request failed with HTTP " + statusCode;
        }
        if (body.contains("\"detail\":")) {
            String marker = "\"detail\":";
            int markerIndex = body.indexOf(marker);
            int startQuote = body.indexOf('"', markerIndex + marker.length());
            if (startQuote >= 0) {
                int endQuote = findStringEnd(body, startQuote + 1);
                return unescapeJson(body.substring(startQuote + 1, endQuote));
            }
        }
        return body;
    }

    private static int findStringEnd(String text, int startIndex) {
        boolean escaped = false;
        for (int index = startIndex; index < text.length(); index++) {
            char current = text.charAt(index);
            if (escaped) {
                escaped = false;
                continue;
            }
            if (current == '\\') {
                escaped = true;
            } else if (current == '"') {
                return index;
            }
        }
        return text.length();
    }

    private static String escapeJson(String value) {
        return value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\r", "\\r")
            .replace("\n", "\\n")
            .replace("\t", "\\t");
    }

    private static String unescapeJson(String value) {
        StringBuilder builder = new StringBuilder();
        boolean escaped = false;
        for (int index = 0; index < value.length(); index++) {
            char current = value.charAt(index);
            if (escaped) {
                switch (current) {
                    case 'n' -> builder.append('\n');
                    case 'r' -> builder.append('\r');
                    case 't' -> builder.append('\t');
                    case '"' -> builder.append('"');
                    case '\\' -> builder.append('\\');
                    default -> builder.append(current);
                }
                escaped = false;
            } else if (current == '\\') {
                escaped = true;
            } else {
                builder.append(current);
            }
        }
        return builder.toString();
    }

    private final class MessageBubble extends RoundedPanel {
        private MessageBubble(String role, String text, Color bubbleColor) {
            super(new GridBagLayout(), bubbleColor, new Color(255, 255, 255, 26));
            setAlignmentX(Component.LEFT_ALIGNMENT);
            setMaximumSize(new Dimension(Integer.MAX_VALUE, Integer.MAX_VALUE));
            setBorder(new EmptyBorder(14, 16, 14, 16));

            GridBagConstraints constraints = new GridBagConstraints();
            constraints.gridx = 0;
            constraints.gridy = 0;
            constraints.anchor = GridBagConstraints.WEST;
            constraints.fill = GridBagConstraints.HORIZONTAL;
            constraints.weightx = 1.0;
            constraints.insets = new Insets(0, 0, 8, 0);

            JLabel roleLabel = new JLabel(role.toUpperCase());
            roleLabel.setForeground(TEXT_MUTED);
            roleLabel.setFont(new Font("Dialog", Font.BOLD, 11));
            add(roleLabel, constraints);

            JTextArea body = new JTextArea(text);
            body.setEditable(false);
            body.setLineWrap(true);
            body.setWrapStyleWord(true);
            body.setOpaque(false);
            body.setForeground(TEXT_PRIMARY);
            body.setFont(new Font("Dialog", Font.PLAIN, 15));
            body.setBorder(BorderFactory.createEmptyBorder());
            body.setFocusable(false);

            constraints.gridy = 1;
            constraints.insets = new Insets(0, 0, 0, 0);
            add(body, constraints);
        }
    }

    private static class RoundedPanel extends JPanel {
        private final Color backgroundColor;
        private final Color borderColor;

        private RoundedPanel(BorderLayout layout, Color backgroundColor, Color borderColor) {
            super(layout);
            this.backgroundColor = backgroundColor;
            this.borderColor = borderColor;
            setOpaque(false);
        }

        private RoundedPanel(GridBagLayout layout, Color backgroundColor, Color borderColor) {
            super(layout);
            this.backgroundColor = backgroundColor;
            this.borderColor = borderColor;
            setOpaque(false);
        }

        @Override
        protected void paintComponent(Graphics graphics) {
            Graphics2D g2 = (Graphics2D) graphics.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setColor(backgroundColor);
            g2.fillRoundRect(0, 0, getWidth() - 1, getHeight() - 1, 26, 26);
            g2.setColor(borderColor);
            g2.setStroke(new BasicStroke(1f));
            g2.drawRoundRect(0, 0, getWidth() - 1, getHeight() - 1, 26, 26);
            g2.dispose();
            super.paintComponent(graphics);
        }
    }

    private static final class GradientPanel extends JPanel {
        @Override
        protected void paintComponent(Graphics graphics) {
            super.paintComponent(graphics);
            Graphics2D g2 = (Graphics2D) graphics.create();
            g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);

            GradientPaint base = new GradientPaint(
                0, 0, new Color(11, 11, 13),
                getWidth(), getHeight(), new Color(18, 18, 22)
            );
            g2.setPaint(base);
            g2.fillRect(0, 0, getWidth(), getHeight());

            g2.setColor(new Color(104, 224, 207, 36));
            g2.fillOval(-80, -120, 340, 340);

            g2.setColor(new Color(249, 212, 143, 32));
            g2.fillOval(getWidth() - 260, -100, 360, 300);

            g2.dispose();
        }
    }
}
