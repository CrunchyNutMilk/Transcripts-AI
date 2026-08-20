"""/game_dictionary — drop-in cog for the Summurizer Discord bot.

Copy this file into the bot's ``cogs/`` folder as ``game_dictionary.py``
and load it like the other cogs. It needs the engine importable
(``pip install -e C:\\dev\\Transcript_AI`` in the bot's environment) and
two pieces of wiring at the ADAPT markers below: the campaign database
path and the guild → campaign-name resolution (both have working env-var
defaults so it runs before any deeper integration).

Commands (everyone can look up; fixing/undoing needs the DM role when
the bot's permission helpers are available, falling back to
manage-messages):

    /game_dictionary lookup name:<text>
    /game_dictionary fix wrong:<text> right:<text> [kind:<choice>]
    /game_dictionary undo wrong:<text>
    /game_dictionary recent

Every fix is recorded as a human decision by the Discord user who made
it — it improves the resolver immediately AND becomes training data for
the local model. Nothing here touches transcripts or the vault.
"""
from __future__ import annotations

import logging
import os

import discord
from discord import app_commands
from discord.ext import commands

from transcripts_ai.dictionary import DictionaryError, GameDictionary
from transcripts_ai.memory import CampaignMemory

logger = logging.getLogger("mybot.game_dictionary")

# --- ADAPT: where the engine's campaign memory lives -----------------------
ENGINE_DB = os.environ.get(
    "ENGINE_DB", r"C:\Users\neill\Documents\engine\engine_memory.sqlite")


def resolve_campaign(interaction: discord.Interaction) -> str:
    """ADAPT: map this guild/channel to its campaign name.

    Default: the ENGINE_CAMPAIGN env var, else the guild's name. If the
    bot's VaultConfigService knows the campaign binding, use that instead.
    """
    configured = os.environ.get("ENGINE_CAMPAIGN")
    if configured:
        return configured
    return interaction.guild.name if interaction.guild else "default"


def _can_edit(interaction: discord.Interaction) -> bool:
    """DM-role gate when the bot's helpers exist; else manage-messages."""
    try:
        from helpers.permissions import has_campaign_dm_role, is_bot_admin
        return bool(is_bot_admin(interaction.user)
                    or has_campaign_dm_role(interaction.user))
    except Exception:
        perms = getattr(interaction.user, "guild_permissions", None)
        return bool(perms and perms.manage_messages)


KIND_CHOICES = [
    app_commands.Choice(name=k, value=k)
    for k in ("pc", "npc", "location", "faction", "item",
              "creature", "quest", "deity", "lore")
]


class GameDictionaryCog(commands.Cog):
    """Table-driven name fixes, straight into campaign memory."""

    group = app_commands.Group(name="game_dictionary",
                               description="Look up and fix campaign names")

    def _dictionary(self, interaction: discord.Interaction
                    ) -> tuple[CampaignMemory, GameDictionary]:
        memory = CampaignMemory(ENGINE_DB)
        return memory, GameDictionary(memory, resolve_campaign(interaction))

    @group.command(name="lookup", description="What is this name in our campaign?")
    async def lookup(self, interaction: discord.Interaction, name: str) -> None:
        memory, dictionary = self._dictionary(interaction)
        try:
            result = dictionary.lookup(name)
        except DictionaryError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        finally:
            memory.close()
        if result.found:
            embed = discord.Embed(
                title=f"{result.canonical} — {result.kind}",
                description=result.description or "*no description yet*")
            if result.aliases:
                embed.add_field(name="Also written as",
                                value=", ".join(result.aliases[:15]))
            embed.set_footer(text=f"status: {result.status}")
            await interaction.response.send_message(embed=embed)
        elif result.suggestions:
            lines = [f'No exact match for "{result.query}". Did you mean:']
            lines += [f"- **{name}** ({score:.0%})"
                      for name, score, _ in result.suggestions]
            await interaction.response.send_message("\n".join(lines),
                                                    ephemeral=True)
        else:
            await interaction.response.send_message(
                f'"{result.query}" is not in the campaign record yet — '
                f"`/game_dictionary fix` can add it.", ephemeral=True)

    @group.command(name="fix",
                   description="Wrong spelling → right name (recorded, reversible)")
    @app_commands.describe(wrong="what the transcript says",
                           right="what it should be",
                           kind="only used when the right name is new")
    @app_commands.choices(kind=KIND_CHOICES)
    async def fix(self, interaction: discord.Interaction, wrong: str,
                  right: str, kind: app_commands.Choice[str] | None = None
                  ) -> None:
        if not _can_edit(interaction):
            await interaction.response.send_message(
                "Fixing names needs the Campaign Manager role.", ephemeral=True)
            return
        memory, dictionary = self._dictionary(interaction)
        actor = f"human:discord:{interaction.user.name}"
        try:
            result = dictionary.fix(wrong, right, actor=actor,
                                    kind=(kind.value if kind else "npc"))
        except DictionaryError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        finally:
            memory.close()
        logger.info("game_dictionary fix by %s: %s -> %s", actor,
                    result.wrong, result.canonical)
        await interaction.response.send_message(f"✅ {result.message}")

    @group.command(name="undo", description="Remove a recorded spelling fix")
    async def undo(self, interaction: discord.Interaction, wrong: str) -> None:
        if not _can_edit(interaction):
            await interaction.response.send_message(
                "Undoing fixes needs the Campaign Manager role.", ephemeral=True)
            return
        memory, dictionary = self._dictionary(interaction)
        actor = f"human:discord:{interaction.user.name}"
        try:
            removed = dictionary.undo(wrong, actor=actor)
        except DictionaryError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        finally:
            memory.close()
        await interaction.response.send_message(
            f'↩️ removed "{wrong}"' if removed
            else f'"{wrong}" was not a recorded spelling.', ephemeral=True)

    @group.command(name="recent", description="Latest dictionary changes")
    async def recent(self, interaction: discord.Interaction) -> None:
        memory, dictionary = self._dictionary(interaction)
        try:
            rows = dictionary.recent(limit=10)
        finally:
            memory.close()
        if not rows:
            await interaction.response.send_message(
                "No dictionary changes yet.", ephemeral=True)
            return
        lines = [f"- `{r['subject']}` — {r['kind'].replace('_', ' ')} "
                 f"({r['actor'].removeprefix('human:')})" for r in rows]
        await interaction.response.send_message(
            "**Recent dictionary changes**\n" + "\n".join(lines))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(GameDictionaryCog(bot))
