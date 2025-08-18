import { Component } from '@angular/core';
import { ResumeSearchService } from '../../services/resume-search.service';

@Component({
  selector: 'app-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent {
  query: string = '';
  loading: boolean = false;
  results: any[] = [];
  summary: string | null = null;

  constructor(private searchService: ResumeSearchService) {}

  onSearch() {
    if (!this.query.trim()) return;

    this.loading = true;
    this.results = [];
    this.summary = null;

    this.searchService.search(this.query).subscribe({
      next: (res) => {
        this.results = res.results;
        this.summary = res.summary;
        this.loading = false;
      },
      error: (err) => {
        console.error('Search error:', err);
        this.loading = false;
      }
    });
  }
}

