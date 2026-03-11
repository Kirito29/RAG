from django.core.management.base import BaseCommand
from core.ai_engine.vector_store import VectorStore


class Command(BaseCommand):

    help = "Build AI vector index"

    def handle(self, *args, **kwargs):

        store = VectorStore()

        store.build_index()

        self.stdout.write(self.style.SUCCESS("Vector index built successfully"))
