using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class Admin
    {
        [Key]
        public int AdminId { get; set; }

        [Required(ErrorMessage = "Email address is required.")]
        [EmailAddress(ErrorMessage = "Invalid email format.")]
        [StringLength(100, ErrorMessage = "Email cannot exceed 100 characters.")]
        public string Email { get; set; }

        [Required(ErrorMessage = "Password is required.")]
        [StringLength(255, ErrorMessage = "Password length exceeded 255 character.")]
        [MinLength(6, ErrorMessage = "Password must be at least 6 characters.")]
        public string PasswordHash { get; set; }

        [Required(ErrorMessage = "Full Name is required.")]
        [StringLength(100, ErrorMessage = "Full Name cannot exceed 100 characters.")]
        public string FullName { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.Now;
    }
}