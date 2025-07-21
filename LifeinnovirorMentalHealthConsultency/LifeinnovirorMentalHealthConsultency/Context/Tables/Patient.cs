using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Security.Policy;
using System.Web;
using LifeinnovirorMentalHealthConsultency.Authorization;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class Patient
    {
        [Key]
        public int PatientId { get; set; }

        [Required(ErrorMessage = "Full name is required.")]
        [StringLength(100)]
        public string FullName { get; set; }

        public DateTime? DateOfBirth { get; set; }

        public string Gender { get; set; }

        [EmailAddress]
        public string Email { get; set; }

        [StringLength(20)]
        public string PhoneNumber { get; set; }

        public string Address { get; set; }

        [StringLength(20)]
        public string EmergencyNumber { get; set; }

        public string MedicalHistoryText { get; set; }

        public decimal? Height { get; set; }

        public decimal? Weight { get; set; }

        [StringLength(50)]
        public string Religion { get; set; }

        public string EducationDetails { get; set; }

        public string Allergies { get; set; }

        [StringLength(50)]
        public string SkinTone { get; set; }

        [StringLength(255)]
        public string ProfilePhotoUrl { get; set; }

        [Required]
        public DateTime CreatedAt { get; set; } = DateTime.Now;

        [Required]
        public DateTime UpdatedAt { get; set; } = DateTime.Now;

        [StringLength(255)]
        public string PasswordHash { get; set; } // deafault should be hashed email address
    }
}