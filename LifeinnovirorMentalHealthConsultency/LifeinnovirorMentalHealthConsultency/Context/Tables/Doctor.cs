using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class Doctor
    {
        [Key]
        public int DoctorId { get; set; }

        [Required(ErrorMessage = "Full Name is required.")]
        [StringLength(100, ErrorMessage = "Full Name cannot exceed 100 characters.")]
        public string FullName { get; set; }

        [Required(ErrorMessage = "Email is required.")]
        [EmailAddress(ErrorMessage = "Invalid email format.")]
        [StringLength(100)]
        public string Email { get; set; }

        [Required(ErrorMessage = "Phone number is required.")]
        [StringLength(20)]
        public string PhoneNumber { get; set; }

        [Required(ErrorMessage = "Password is required.")]
        [StringLength(255)]
        [MinLength(6, ErrorMessage = "Password must be at least 6 characters.")]
        public string PasswordHash { get; set; }

        [Required(ErrorMessage = "Security question is required.")]
        public string SecurityQuestion { get; set; }

        [Required(ErrorMessage = "Security answer is required.")]
        public string SecurityAnswerHash { get; set; }

        public string Qualifications { get; set; }

        public string ExperienceSummary { get; set; }

        public int? YearsOfExperience { get; set; }
        public int minimumCancelTime { get; set; } = 1; 

        [StringLength(255)]
        public string ProfilePhotoUrl { get; set; }

        [StringLength(20)]
        [RegularExpression("Pending|Approved|Rejected",
            ErrorMessage = "Status must be Pending, Approved, or Rejected")]
        public string Status { get; set; } = "Pending";

        [Required]
        public DateTime CreatedAt { get; set; } = DateTime.Now;

        [Required]
        public DateTime UpdatedAt { get; set; } = DateTime.Now;
        public virtual ICollection<DoctorTimeSlot> DoctorTimeSlots { get; set; }

    }

}