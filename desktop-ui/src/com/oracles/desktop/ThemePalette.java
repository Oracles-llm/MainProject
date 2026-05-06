package com.oracles.desktop;

import java.awt.Color;

final class ThemePalette {
    final Color background;
    final Color foreground;
    final Color card;
    final Color cardForeground;
    final Color primary;
    final Color primaryForeground;
    final Color secondary;
    final Color secondaryForeground;
    final Color muted;
    final Color mutedForeground;
    final Color accent;
    final Color accentForeground;
    final Color border;
    final Color input;
    final Color ring;
    final Color sidebar;
    final Color sidebarForeground;
    final Color sidebarAccent;
    final Color sidebarAccentForeground;
    final Color sidebarBorder;
    final Color assistantBubble;
    final Color overlay;
    final Color destructive;

    private ThemePalette(
        Color background,
        Color foreground,
        Color card,
        Color cardForeground,
        Color primary,
        Color primaryForeground,
        Color secondary,
        Color secondaryForeground,
        Color muted,
        Color mutedForeground,
        Color accent,
        Color accentForeground,
        Color border,
        Color input,
        Color ring,
        Color sidebar,
        Color sidebarForeground,
        Color sidebarAccent,
        Color sidebarAccentForeground,
        Color sidebarBorder,
        Color assistantBubble,
        Color overlay,
        Color destructive
    ) {
        this.background = background;
        this.foreground = foreground;
        this.card = card;
        this.cardForeground = cardForeground;
        this.primary = primary;
        this.primaryForeground = primaryForeground;
        this.secondary = secondary;
        this.secondaryForeground = secondaryForeground;
        this.muted = muted;
        this.mutedForeground = mutedForeground;
        this.accent = accent;
        this.accentForeground = accentForeground;
        this.border = border;
        this.input = input;
        this.ring = ring;
        this.sidebar = sidebar;
        this.sidebarForeground = sidebarForeground;
        this.sidebarAccent = sidebarAccent;
        this.sidebarAccentForeground = sidebarAccentForeground;
        this.sidebarBorder = sidebarBorder;
        this.assistantBubble = assistantBubble;
        this.overlay = overlay;
        this.destructive = destructive;
    }

    static ThemePalette light() {
        return new ThemePalette(
            new Color(255, 255, 255),
            new Color(24, 31, 45),
            new Color(255, 255, 255),
            new Color(24, 31, 45),
            new Color(33, 41, 55),
            new Color(248, 250, 252),
            new Color(241, 245, 249),
            new Color(33, 41, 55),
            new Color(241, 245, 249),
            new Color(100, 116, 139),
            new Color(241, 245, 249),
            new Color(33, 41, 55),
            new Color(226, 232, 240),
            new Color(226, 232, 240),
            new Color(148, 163, 184),
            new Color(250, 250, 250),
            new Color(24, 31, 45),
            new Color(241, 245, 249),
            new Color(33, 41, 55),
            new Color(226, 232, 240),
            new Color(245, 248, 251),
            new Color(255, 255, 255, 210),
            new Color(220, 38, 38)
        );
    }

    static ThemePalette dark() {
        return new ThemePalette(
            new Color(18, 24, 38),
            new Color(248, 250, 252),
            new Color(33, 41, 55),
            new Color(248, 250, 252),
            new Color(226, 232, 240),
            new Color(33, 41, 55),
            new Color(51, 65, 85),
            new Color(248, 250, 252),
            new Color(51, 65, 85),
            new Color(148, 163, 184),
            new Color(51, 65, 85),
            new Color(248, 250, 252),
            new Color(255, 255, 255, 24),
            new Color(255, 255, 255, 38),
            new Color(113, 113, 122),
            new Color(33, 41, 55),
            new Color(248, 250, 252),
            new Color(51, 65, 85),
            new Color(248, 250, 252),
            new Color(255, 255, 255, 24),
            new Color(61, 76, 98),
            new Color(16, 22, 36, 230),
            new Color(248, 113, 113)
        );
    }
}
